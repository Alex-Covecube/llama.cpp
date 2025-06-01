#include "ggml.h"
#include "ggml-numa.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "spsc-queue.h"

#include <cinttypes>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <thread>
#include <deque>
#include <condition_variable>
#include <future>
#include <semaphore>
#include <queue>

#pragma region Async structures

#define NUMA_ASYNC 1
#define NUMA_EVENTS (NUMA_ASYNC && 0)

#pragma region Manual Reset Event

class manual_reset_event {  
public:
    manual_reset_event(bool signaled = false)
        : signaled_(signaled) {}

    void set() {
        std::lock_guard<std::mutex> lock(m_);
        signaled_ = true;
        cv_.notify_all();
    }

    void reset() {
        std::lock_guard<std::mutex> lock(m_);
        signaled_ = false;
    }

    void wait() {
        std::unique_lock<std::mutex> lock(m_);
        cv_.wait(lock, [this] { return signaled_; });
    }

private:
    std::mutex m_;
    std::condition_variable cv_;
    bool signaled_;
};

#pragma endregion

enum class numa_task_type {
    SET, GET, CPY,
    GRAPH_PLAN_COMPUTE,
    GRAPH_COMPUTE,
    BARRIER,
};

struct numa_task {
    numa_task_type                       type;
    // --- for SET / GET ---------------------------------------------------------
    struct ggml_tensor                 * t_dst  { nullptr };
    const struct ggml_tensor           * t_src  { nullptr };
    void                               * h_dst  { nullptr };
    const void                         * h_src  { nullptr };
    ggml_backend_t                       b_src  { nullptr };
    ggml_backend_t                       b_dst  { nullptr };
    size_t                               off    { 0 };
    size_t                               size   { 0 };
    // --- for graph -------------------------------------------------------------
    ggml_backend_graph_plan_t            plan   { nullptr };
    struct ggml_cgraph                 * cgraph { nullptr };
    // --------------------------------------------------------------------------
    std::binary_semaphore              * synchronize   { nullptr }; // From ggml_backend_numa_synchronize
    manual_reset_event                 * event_record  { nullptr }; // From ggml_backend_numa_event_record
    bool                                 event_record_skip { false }; // Skip waiting on event_record
    bool                                 event_record_wait { false }; // Wait for event_record to be signaled
};

#define NUMA_EVENT_RECORDS 2

struct numa_event {
    manual_reset_event          record[NUMA_EVENT_RECORDS] { false };        // event that is signaled when the event is recorded
    int                         record_index { 0 };         // index of the record to use
    numa_task *                 task_record { nullptr };    // task that contains the record semaphore
    std::vector<numa_task *>    tasks;                      // tasks that are associated with this event
};

#pragma endregion

#pragma region Contexts

// contexts
struct ggml_backend_numa_reg_context {
    ggml_backend_reg_t backend_reg_cpu;
    std::recursive_mutex devices_mutex;
    std::unordered_map<size_t, ggml_backend_dev_t> devices;
    bool devices_initialized = false;
};

#define GGML_NUMA_QUEUE_SIZE 1024

// Async context for NUMA device
struct ggml_backend_numa_backend_dev_async_context {
    std::queue<numa_task*>  queue { };
    std::binary_semaphore   has_work { 1 };
    std::thread             worker;
    std::atomic<bool>       stop = false;
};

struct ggml_backend_numa_backend_dev_context {
    std::recursive_mutex mutex;

    ggml_backend_dev_t backend_dev_cpu;

    ggml_backend_buffer_type_t buffer_type;
    ggml_backend_t backend;

    std::string name;
    std::string description;

    int numa_node = -1;
    uint32_t cpu_count = 0;
    ggml_threadpool_t threadpool = NULL;
    
    ggml_backend_numa_backend_dev_async_context async;
};

struct ggml_backend_numa_backend_context {
    ggml_backend_t backend_cpu;

    std::string name;
};

struct ggml_backend_numa_buffer_type_context {
    ggml_backend_buffer_type_t buffer_type_cpu;

    std::string name;
};

#pragma endregion

#pragma region Async

static std::mutex numa_mutex;

static void numa_worker_loop(ggml_backend_device * numa_device) {
    
    auto ctx = (ggml_backend_numa_backend_dev_context *) numa_device->context;
    auto backend = (ggml_backend_t) numa_device->iface.init_backend(numa_device, nullptr);
    auto backend_ctx = (ggml_backend_numa_backend_context *) backend->context;
    auto backend_cpu = (ggml_backend_t) backend_ctx->backend_cpu;

    GGML_LOG_DEBUG("%s: [NUMA %d] started worker\n", __func__, ctx->numa_node);

    while (true) {
        GGML_LOG_DEBUG("%s: [NUMA %d] waiting for task...\n", __func__, ctx->numa_node);
        while (true) {
            numa_task * task_ptr;
            {
                std::unique_lock<std::mutex> lock(numa_mutex);
                if (ctx->async.queue.empty()) {
                    break;
                }

                task_ptr = ctx->async.queue.front();
                ctx->async.queue.pop();
            }

            auto & task = *task_ptr;

            GGML_LOG_DEBUG("%s: [NUMA %d] got task %d\n", __func__, ctx->numa_node, (int)task.type);

            switch (task.type) {
                case numa_task_type::SET: {
                    ggml_backend_buffer_t t_dst_buf = task.t_dst->view_src ? task.t_dst->view_src->buffer : task.t_dst->buffer;
                    t_dst_buf->iface.set_tensor(t_dst_buf, task.t_dst, task.h_src, task.off, task.size);
                    break;
                }
                case numa_task_type::GET: {
                    ggml_backend_buffer_t t_src_buf = task.t_src->view_src ? task.t_src->view_src->buffer : task.t_src->buffer;
                    t_src_buf->iface.get_tensor(t_src_buf, task.t_src, task.h_dst, task.off, task.size);
                    break;
                }
                case numa_task_type::CPY: {
                    if (task.b_src != task.b_dst) {
                        // We're copying between different backends, so we need to synchronize the source backend first
                        // We are the destination backend, so we don't need to synchronize ourself
                        ggml_backend_synchronize(task.b_src);
                    }

                    ggml_backend_tensor_copy(const_cast<ggml_tensor *>(task.t_src), task.t_dst);
                    break;
                }
                case numa_task_type::GRAPH_PLAN_COMPUTE: {
                    backend_cpu->iface.graph_plan_compute(backend_cpu, task.plan);
                    break;
                }
                case numa_task_type::GRAPH_COMPUTE: {
                    backend_cpu->iface.graph_compute(backend_cpu, task.cgraph);
                    break;
                }
                case numa_task_type::BARRIER: {
                    /* nothing extra to do – we are the barrier */
                    break;
                }
            }

            if (task.event_record != nullptr || task.synchronize != nullptr) {
                if (task.event_record != nullptr) {
                    if (task.event_record_wait){
                        // Wait for another work thread to reach a certain point
                        auto t1 = std::chrono::high_resolution_clock::now();
                        GGML_LOG_DEBUG("%s: [NUMA %d] task %d waiting for record event (record=%p)...\n", __func__, ctx->numa_node, (int)task.type, (void*)task.event_record);

                        task.event_record->wait();

                        auto t2 = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double, std::milli> wait_duration = t2 - t1;
                        GGML_LOG_DEBUG("%s: [NUMA %d] task %d wait done (%.4f ms) (record=%p)\n", __func__, ctx->numa_node, (int)task.type, wait_duration.count(), (void*)task.event_record);
                    }
                    else {
                        if (!task.event_record_skip){
                            // Set the event to signal to another work thread that we've reached a certain point
                            GGML_LOG_DEBUG("%s: [NUMA %d] setting task %d record event (record=%p)\n", __func__, ctx->numa_node, (int)task.type, (void*)task.event_record);
                            task.event_record->set();
                        } else {
                            GGML_LOG_DEBUG("%s: [NUMA %d] task %d record event skipped (record=%p)\n", __func__, ctx->numa_node, (int)task.type, (void*)task.event_record);
                        }
                    }
                }

                if (task.synchronize != nullptr) {
                    // Inform the producer that the task is done
                    GGML_LOG_DEBUG("%s: [NUMA %d] task %d synchronize set\n", __func__, ctx->numa_node, (int)task.type);
                    task.synchronize->release();
                }
            } else {
                // Clean up the task
                delete task_ptr;
            } 
        }

        if (ctx->async.stop) {
            GGML_LOG_DEBUG("%s: [NUMA %d] stopping worker thread\n", __func__, ctx->numa_node);
            break;
        }

        // wait for new tasks
        ctx->async.has_work.acquire();
    }
}

static void numa_push_task(ggml_backend_numa_backend_dev_context * ctx,
                           numa_task * task) {

    GGML_LOG_DEBUG("%s: [NUMA %d] pushing new task %d and notifying worker thread\n", __func__, ctx->numa_node, (int)task->type);

    std::unique_lock<std::mutex> lock(numa_mutex);

    ctx->async.queue.push(task);
    ctx->async.has_work.release();
}

#pragma region Events

// Note that all event functions are called from the main thread, so they do not need to be thread-safe.

// device->iface.event_new
static ggml_backend_event_t ggml_backend_numa_event_new(ggml_backend_dev_t dev) {
    auto dev_ctx = (ggml_backend_numa_backend_dev_context *) dev->context;

    auto r = new ggml_backend_event();
    r->device = dev;
    r->context = new numa_event();

    GGML_LOG_DEBUG("%s: [NUMA %d] created a new event for NUMA device (event=%p) \n", __func__, dev_ctx->numa_node, (void*)r->context);

    return r;
}

// device->iface.event_free
static void ggml_backend_numa_event_free(ggml_backend_dev_t dev,
                                         ggml_backend_event_t ev) {
    auto dev_ctx = (ggml_backend_numa_backend_dev_context *) dev->context;

    if (ev == nullptr) {
        GGML_LOG_DEBUG("%s: [NUMA %d] event is nullptr, nothing to free\n", __func__, dev_ctx->numa_node);
        return;
    }

    GGML_LOG_DEBUG("%s: [NUMA %d] freeing event (event=%p)\n", __func__, dev_ctx->numa_node, (void*)ev->context);

    // Free the tasks associated with the event
    auto * nev = (numa_event *) ev->context;
    if (!nev->tasks.empty()) {
        GGML_LOG_DEBUG("%s: freeing %d tasks\n", __func__, (int)nev->tasks.size());
        for (auto * task : nev->tasks) {
            if (task) {
                delete task;
            }
        }
    }

    delete nev;
    delete (numa_event *) ev;
}

// device->iface.event_synchronize
// Wait for a recorded event to be signaled.
static void ggml_backend_numa_event_synchronize(ggml_backend_dev_t dev,
                                                ggml_backend_event_t ev) {
    
    auto dev_ctx = (ggml_backend_numa_backend_dev_context *) dev->context;
    auto * nev = (numa_event *) ev->context;

    if (!nev) {
        GGML_LOG_DEBUG("%s: [NUMA %d] event is nullptr, nothing to wait for\n", __func__, dev_ctx->numa_node);
        return;
    }

    if (nev->task_record == nullptr) {
        GGML_LOG_DEBUG("%s: [NUMA %d] event is not recorded, nothing to wait for (event=%p)\n", __func__, dev_ctx->numa_node, (void*)nev);
        return;
    }

    if (nev->task_record->event_record_skip) {
        GGML_LOG_DEBUG("%s: [NUMA %d] record event is marked for skipping, nothing to wait for (event=%p, record=%p)\n", __func__, dev_ctx->numa_node, (void*)nev, (void*)&nev->record[nev->record_index]);
        return;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    GGML_LOG_DEBUG("%s: [NUMA %d] synchronized waiting for event (event=%p, record=%p)...\n", __func__, dev_ctx->numa_node, (void*)nev, (void*)&nev->record[nev->record_index]);

    nev->record[nev->record_index].wait();

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> wait_duration = t2 - t1;
    GGML_LOG_DEBUG("%s: [NUMA %d] synchronized wait done (%.4f ms) for event (event=%p, record=%p)\n", __func__, dev_ctx->numa_node, wait_duration.count(), (void*)nev, (void*)&nev->record[nev->record_index]);

    GGML_UNUSED(dev);
}

// backend->iface.event_record
// Insert the event into the work queue so that it will be signaled all previous tasks are done.
static void ggml_backend_numa_event_record(ggml_backend_t backend,
                                           ggml_backend_event_t ev) {
    
    auto ctx = (ggml_backend_numa_backend_dev_context *) backend->device->context;
    auto * nev = (numa_event *) ev->context;

    if (!nev) {
        GGML_LOG_DEBUG("%s: [NUMA %d] event is nullptr, nothing to record\n", __func__, ctx->numa_node);
        return;
    }

    auto task_record_previous = nev->task_record;

    auto next_record_index = (nev->record_index + 1) % NUMA_EVENT_RECORDS;

    GGML_LOG_DEBUG("%s: [NUMA %d] recording requested for event (event=%p, record=%p)\n", __func__, ctx->numa_node, (void*)nev, (void*)&nev->record[next_record_index]);

    numa_task * task = new numa_task();
    task->type = numa_task_type::BARRIER;
    task->event_record = &nev->record[next_record_index]; // The work thread will signal another work thread that it has processed all previous tasks

    // Reset the event to be signaled later
    nev->record[next_record_index].reset();

    // Store the task in the event for later cleanup
    nev->tasks.push_back(task);
    nev->task_record = task;
    nev->record_index = next_record_index;

    numa_push_task(ctx, task);

    if (task_record_previous != nullptr) {
        // Skip the previously recorded event
        // Note that whether it will actually be skipped depends on timing, so this is not a guarantee
        GGML_LOG_DEBUG("%s: [NUMA %d]   marking previous event for skipping\n", __func__, ctx->numa_node);
        task_record_previous->event_record_skip = true;
        return;
    }
}

// backend->iface.event_wait
// Insert the event into the work queue so that it will cause the work queue to wait until the event is signaled.
static void ggml_backend_numa_event_wait(ggml_backend_t backend,
                                         ggml_backend_event_t ev) {
    auto ctx = (ggml_backend_numa_backend_dev_context *) backend->device->context;
    auto * nev = (numa_event *) ev->context;

    if (!nev) {
        GGML_LOG_DEBUG("%s: [NUMA %d] event is nullptr, nothing to wait for\n", __func__, ctx->numa_node);
        return;
    }

    if (nev->task_record == nullptr) {
        GGML_LOG_DEBUG("%s: [NUMA %d] event is not recorded, nothing to wait for (event=%p)\n", __func__, ctx->numa_node, (void*)nev);
        return;
    }

    if (nev->task_record->event_record_skip){
        GGML_LOG_DEBUG("%s: [NUMA %d] record event is marked for skipping, nothing to wait for (event=%p, record=%p)\n", __func__, ctx->numa_node, (void*)nev, (void*)&nev->record[nev->record_index]);
        return;
    }

    GGML_LOG_DEBUG("%s: [NUMA %d] recording wait for event (event=%p, record=%p)\n", __func__, ctx->numa_node, (void*)nev, (void*)&nev->record[nev->record_index]);

    // Create a task that will wait for the event to be signaled
    auto task = new numa_task();
    task->type = numa_task_type::BARRIER;
    task->event_record = &nev->record[nev->record_index];   // The work thread will wait for another work thread to reach a certain point
    task->event_record_wait = true;                         // Wait for the event to be signaled

    // Store the task in the event for later cleanup
    nev->tasks.push_back(task);

    numa_push_task(ctx, task);
}

#pragma endregion

#pragma endregion

#pragma region NUMA data structures

static ggml_guid_t ggml_backend_numa_guid() {
    static ggml_guid guid = {0x1d, 0xbc, 0x15, 0x41, 0xab, 0x32, 0x41, 0x83, 0xa6, 0xe7, 0x64, 0x14, 0xd8, 0xf6, 0xc9, 0x18}; //{1DBC1541-AB32-4183-A6E7-6414D8F6C918}
    return &guid;
}

#pragma endregion

static const char * ggml_backend_numa_name(ggml_backend_t backend) {
    auto numa_ctx = (ggml_backend_numa_backend_context *)backend->context;

    return numa_ctx->name.c_str();
}

static void ggml_backend_numa_free(ggml_backend_t backend) {
    auto numa_ctx = (ggml_backend_numa_backend_context *)backend->context;
    auto dev_ctx  = (ggml_backend_numa_backend_dev_context *) backend->device->context;

    GGML_LOG_DEBUG("%s: [NUMA %d] freeing backend (stopping worker thread)\n", __func__, dev_ctx->numa_node);

    // ask the worker to exit
    dev_ctx->async.stop = true;
    dev_ctx->async.has_work.release();

    if (dev_ctx->async.worker.joinable())
        dev_ctx->async.worker.join();

    auto backend_cpu = (ggml_backend_t)numa_ctx->backend_cpu;
    if (backend_cpu != NULL) {
        backend_cpu->iface.free(backend_cpu);
    }

    delete numa_ctx;
    delete backend;
}

static void ggml_backend_numa_synchronize(ggml_backend_t backend) {
#if NUMA_ASYNC
    auto dev_ctx = (ggml_backend_numa_backend_dev_context *) backend->device->context;

    GGML_LOG_DEBUG("%s: [NUMA %d] synchronizing backend...\n", __func__, dev_ctx->numa_node);

    std::binary_semaphore done_sem(0);

    auto task = new numa_task();
    task->type = numa_task_type::BARRIER;
    task->synchronize = &done_sem; // The worker will signal this semaphore when it finishes

    numa_push_task(dev_ctx, task);

    // wait until the worker finishes everything
    done_sem.acquire();

    delete task;

    GGML_LOG_DEBUG("%s: [NUMA %d] backend synchronized\n", __func__, dev_ctx->numa_node);
#else
    // Forward the call to the CPU backend
    
    auto numa_ctx = (ggml_backend_numa_backend_context *)backend->context;
    auto backend_cpu = (ggml_backend_t)numa_ctx->backend_cpu;
    if (backend_cpu != NULL) {
        backend_cpu->iface.synchronize(backend_cpu);
    }
    else {
        GGML_LOG_ERROR("%s: synchronize called on a non-CPU backend\n", __func__);
    }
#endif
}

static ggml_backend_graph_plan_t ggml_backend_numa_graph_plan_create(ggml_backend_t backend, const struct ggml_cgraph * cgraph) {
    auto numa_ctx = (ggml_backend_numa_backend_context *)backend->context;

    auto backend_cpu = (ggml_backend_t)numa_ctx->backend_cpu;
    if (backend_cpu != NULL) {
        return backend_cpu->iface.graph_plan_create(backend_cpu, cgraph);
    }
    else {
        GGML_LOG_ERROR("%s: graph_plan_create called on a non-CPU backend\n", __func__);
        return NULL;
    }
}

static void ggml_backend_numa_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    auto numa_ctx = (ggml_backend_numa_backend_context *)backend->context;

    auto backend_cpu = (ggml_backend_t)numa_ctx->backend_cpu;
    if (backend_cpu != NULL) {
        backend_cpu->iface.graph_plan_free(backend_cpu, plan);
    }
    else {
        GGML_LOG_ERROR("%s: graph_plan_free called on a non-CPU backend\n", __func__);
    }
}

static enum ggml_status ggml_backend_numa_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    auto numa_ctx = (ggml_backend_numa_backend_context *)backend->context;

    auto backend_cpu = (ggml_backend_t)numa_ctx->backend_cpu;
    if (backend_cpu != NULL) {
        return backend_cpu->iface.graph_plan_compute(backend_cpu, plan);
    }
    else {
        GGML_LOG_ERROR("%s: graph_plan_compute called on a non-CPU backend\n", __func__);
        return GGML_STATUS_FAILED;
    }
}

static void ggml_backend_numa_graph_compute_async(ggml_backend_t backend,
                                                  struct ggml_cgraph * cg) {
    auto dev_ctx = (ggml_backend_numa_backend_dev_context *) backend->device->context;

    auto task = new numa_task();

    task->type = numa_task_type::GRAPH_COMPUTE;
    task->cgraph = cg;

    numa_push_task(dev_ctx, task);
}

static enum ggml_status ggml_backend_numa_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
#if NUMA_ASYNC
    ggml_backend_numa_graph_compute_async(backend, cgraph);
    return GGML_STATUS_SUCCESS;
#else
    // Forward the call to the CPU backend
    auto numa_ctx = (ggml_backend_numa_backend_context *)backend->context;

    auto backend_cpu = (ggml_backend_t)numa_ctx->backend_cpu;
    if (backend_cpu != NULL) {
        return backend_cpu->iface.graph_compute(backend_cpu, cgraph);
    }
    else {
        GGML_LOG_ERROR("%s: graph_compute called on a non-CPU backend\n", __func__);
        return GGML_STATUS_FAILED;
    }
#endif
}

static void ggml_backend_numa_set_tensor_async(ggml_backend_t backend,
                                               struct ggml_tensor * t,
                                               const void * data,
                                               size_t off, size_t size) {
    auto dev_ctx = (ggml_backend_numa_backend_dev_context *) backend->device->context;

    auto task = new numa_task();

    task->type = numa_task_type::SET;
    task->t_dst = t;
    task->h_src = data;
    task->off = off;
    task->size = size;

    numa_push_task(dev_ctx, task);
}

static void ggml_backend_numa_get_tensor_async(ggml_backend_t backend,
                                               const struct ggml_tensor * t,
                                               void * data,
                                               size_t off, size_t size) {
    auto dev_ctx = (ggml_backend_numa_backend_dev_context *) backend->device->context;

    auto task = new numa_task();

    task->type = numa_task_type::GET;
    task->t_src = t;
    task->h_dst = data;
    task->off = off;
    task->size = size;

    numa_push_task(dev_ctx, task);
}

static bool ggml_backend_numa_cpy_tensor_async(ggml_backend_t backend_src,
                                               ggml_backend_t backend_dst, 
                                               const struct ggml_tensor * src, 
                                               struct ggml_tensor * dst) {

                                                // Ensure that the backends are not null
    if (backend_src == nullptr || backend_dst == nullptr) {
        GGML_LOG_ERROR("%s: source or destination backend is NULL\n", __func__);
        return false;
    }

    // Ensure that the source and destination backends are ours by comparing their GUIDs
    if (!ggml_guid_matches(backend_src->guid, ggml_backend_numa_guid()) || !ggml_guid_matches(backend_dst->guid, ggml_backend_numa_guid())) {
        GGML_LOG_DEBUG("%s: source or destination backend is not a NUMA backend\n", __func__);
        return false;
    }

    auto dev_ctx = (ggml_backend_numa_backend_dev_context *) backend_dst->device->context;

    auto task = new numa_task();

    task->type = numa_task_type::CPY;
    task->t_dst = dst;
    task->t_src = src;
    task->b_src = backend_src;
    task->b_dst = backend_dst;

    numa_push_task(dev_ctx, task);
    return true;

    GGML_UNUSED(backend_src);
}


static ggml_backend_i ggml_backend_numa_i = {
    /* .get_name                = */ ggml_backend_numa_name,
    /* .free                    = */ ggml_backend_numa_free,
#if NUMA_ASYNC
    /* .set_tensor_async        = */ ggml_backend_numa_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_numa_get_tensor_async,
    /* .cpy_tensor_async        = */ ggml_backend_numa_cpy_tensor_async,
#else
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
#endif
    /* .synchronize             = */ ggml_backend_numa_synchronize,
    /* .graph_plan_create       = */ ggml_backend_numa_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_numa_graph_plan_free,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ ggml_backend_numa_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_numa_graph_compute,
#if NUMA_EVENTS
    /* .event_record            = */ ggml_backend_numa_event_record,
    /* .event_wait              = */ ggml_backend_numa_event_wait,
#else
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
#endif
};

// device interface

static const char * ggml_backend_numa_device_get_name(ggml_backend_dev_t dev) {
    auto ctx = (ggml_backend_numa_backend_dev_context *)dev->context;

    return ctx->name.c_str();
}

static const char * ggml_backend_numa_device_get_description(ggml_backend_dev_t dev) {
    auto ctx = (ggml_backend_numa_backend_dev_context *)dev->context;

    return ctx->description.c_str();
}

static void ggml_backend_numa_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    auto ctx = (ggml_backend_numa_backend_dev_context *)dev->context;

    // Forward the call to the CPU backend
    ctx->backend_dev_cpu->iface.get_memory(ctx->backend_dev_cpu, free, total);
}

static enum ggml_backend_dev_type ggml_backend_numa_device_get_type(ggml_backend_dev_t dev) {
    // Pretend to be a GPU device for GPU layer compatibility
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_numa_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_numa_device_get_name(dev);
    props->description = ggml_backend_numa_device_get_description(dev);
    props->type        = ggml_backend_numa_device_get_type(dev);
    ggml_backend_numa_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
#if NUMA_ASYNC
        /* .async                 = */ true,
#else
        /* .async                 = */ false,
#endif
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
#if NUMA_EVENTS
        /* .events                = */ true,
#else
        /* .events                = */ false,
#endif
    };
}

static ggml_backend_t ggml_backend_numa_device_init(ggml_backend_dev_t dev, const char * params) {
    auto ctx = (ggml_backend_numa_backend_dev_context *)dev->context;

    std::lock_guard<std::recursive_mutex> lock(ctx->mutex);
    
    if (ctx->backend != NULL) {
        // already initialized
        return ctx->backend;
    }

    // Forward the call to the CPU backend
    auto backend_cpu = ctx->backend_dev_cpu->iface.init_backend(ctx->backend_dev_cpu, params);
    if (backend_cpu == NULL) {
        GGML_LOG_ERROR("%s: Failed to initialize CPU backend\n", __func__);
        return NULL;
    }

    auto ctx_backend = new ggml_backend_numa_backend_context {
        /* .backend_cpu = */ backend_cpu,
        /* .name        = */ ctx->name,
    };

    if (ctx_backend == NULL) {
        GGML_LOG_ERROR("%s: Failed to create NUMA backend context\n", __func__);
        return NULL;
    }

    auto backend = new ggml_backend {
        /* .guid      = */ ggml_backend_numa_guid(),
        /* .interface = */ ggml_backend_numa_i,
        /* .device    = */ dev,
        /* .context   = */ ctx_backend
    };
    
    if (backend == NULL) {
        GGML_LOG_ERROR("%s: Failed to create NUMA backend\n", __func__);
        delete ctx_backend;
        return NULL;
    }

    ctx->backend = backend;

    return backend;

    GGML_UNUSED(params);
}

static const char * ggml_backend_numa_buffer_type_name(ggml_backend_buffer_type_t buft) {
    auto ctx = (ggml_backend_numa_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_numa_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto buft_name = buft->iface.get_name(buft);
    
    auto ctx = (ggml_backend_numa_buffer_type_context *)buft->context;
    auto ctx_backend_dev = (ggml_backend_numa_backend_dev_context*)buft->device->context;

    // Forward the call to the CPU backend
    auto buffer = ctx->buffer_type_cpu->iface.alloc_buffer(ctx->buffer_type_cpu, size);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: Failed to allocate buffer of size %zu for NUMA node %d\n", __func__, size, ctx_backend_dev->numa_node);
        return NULL;
    }

    // The allocated buffer should belong to the NUMA buffer type
    buffer->buft = buft;

    GGML_LOG_DEBUG("%s: Allocated buffer of size %zu for NUMA node %d (buft_name: %s, buft: %p, buf->buft: %p)\n", __func__, size, ctx_backend_dev->numa_node, buft_name, (void*)buft, (void*)buffer->buft);

    return buffer;
}

static size_t ggml_backend_numa_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    auto ctx = (ggml_backend_numa_buffer_type_context *)buft->context;

    // Forward the call to the CPU backend
    return ctx->buffer_type_cpu->iface.get_alignment(ctx->buffer_type_cpu);
}

static bool ggml_backend_numa_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    auto ctx = (ggml_backend_numa_buffer_type_context *)buft->context;

    // Forward the call to the CPU backend
    return ctx->buffer_type_cpu->iface.is_host(ctx->buffer_type_cpu);
}

static ggml_backend_buffer_type_i ggml_backend_numa_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_numa_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_numa_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_numa_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL,
    /* .get_alloc_size   = */ NULL,
    /* .is_host          = */ ggml_backend_numa_buffer_type_is_host,
};

static ggml_backend_buffer_type_t ggml_backend_numa_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto ctx = (ggml_backend_numa_backend_dev_context *)dev->context;

    std::lock_guard<std::recursive_mutex> lock(ctx->mutex);

    if (ctx->buffer_type != NULL) {
        return ctx->buffer_type;
    }

    // Get the CPU backend buffer type
    auto buffer_type_cpu = ctx->backend_dev_cpu->iface.get_buffer_type(ctx->backend_dev_cpu);
    if (buffer_type_cpu == NULL) {
        GGML_LOG_ERROR("%s: Failed to get CPU backend buffer type\n", __func__);
        return NULL;
    }

    auto ctx_buffer_type = new ggml_backend_numa_buffer_type_context {
        /* .buffer_type_cpu = */ buffer_type_cpu,
        /* .name            = */ "NUMA" + std::to_string(ctx->numa_node),
    };

    if (ctx_buffer_type == NULL) {
        GGML_LOG_ERROR("%s: Failed to create NUMA buffer type context\n", __func__);
        return NULL;
    }

    auto buft = new ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_numa_buffer_type_interface,
        /* .device  = */ dev,
        /* .context = */ ctx_buffer_type
    };
    
    if (buft == NULL) {
        GGML_LOG_ERROR("%s: Failed to create NUMA buffer type\n", __func__);
        delete ctx_buffer_type;
        return NULL;
    }
    
    ctx->buffer_type = buft;

    return buft;
}

static bool ggml_backend_numa_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    auto ctx = (ggml_backend_numa_backend_dev_context *)dev->context;

    // Forward the call to the CPU backend
    bool supported = ctx->backend_dev_cpu->iface.supports_op(ctx->backend_dev_cpu, op);
    
    GGML_LOG_DEBUG("%s: supports_op (op: %d) for NUMA node %d returned %d\n", __func__, op->op, ctx->numa_node, supported);
    
    return supported;
}

static bool ggml_backend_numa_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (dev == nullptr || buft == nullptr || buft->device == nullptr) {
        GGML_LOG_DEBUG("%s: Null pointer encountered in supports_buft\n", __func__);
        return false;
    }

    auto ctx = (ggml_backend_numa_backend_dev_context *)dev->context;

    auto buft_name = buft->iface.get_name ? buft->iface.get_name(buft) : "unknown";

    auto r = buft->device == dev;

    GGML_LOG_DEBUG("%s: supports_buft (buft: %s) for NUMA node %d returned %s\n", __func__, buft_name, ctx ? ctx->numa_node : -1, r ? "true" : "false");

    return r;
}

static const struct ggml_backend_device_i ggml_backend_numa_device_i = {
    /* .get_name             = */ ggml_backend_numa_device_get_name,
    /* .get_description      = */ ggml_backend_numa_device_get_description,
    /* .get_memory           = */ ggml_backend_numa_device_get_memory,
    /* .get_type             = */ ggml_backend_numa_device_get_type,
    /* .get_props            = */ ggml_backend_numa_device_get_props,
    /* .init_backend         = */ ggml_backend_numa_device_init,
    /* .get_buffer_type      = */ ggml_backend_numa_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_numa_device_supports_op,
    /* .supports_buft        = */ ggml_backend_numa_device_supports_buft,
    /* .offload_op           = */ NULL,
#if NUMA_EVENTS
    /* .event_new            = */ ggml_backend_numa_event_new,
    /* .event_free           = */ ggml_backend_numa_event_free,
    /* .event_synchronize    = */ ggml_backend_numa_event_synchronize,
#else
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
#endif    
};

// backend reg interface

static const char * ggml_backend_numa_reg_get_name(ggml_backend_reg_t reg) {
    return GGML_BACKEND_NUMA_NAME;

    GGML_UNUSED(reg);
}

static size_t ggml_backend_numa_reg_get_device_count(ggml_backend_reg_t reg) {

    auto ctx = (ggml_backend_numa_reg_context *)reg->context;

    std::lock_guard<std::recursive_mutex> lock(ctx->devices_mutex);

    GGML_LOG_DEBUG("%s: get_device_count: %zu\n", __func__, ctx->devices.size());

    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_numa_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto ctx = (ggml_backend_numa_reg_context *)reg->context;

    std::lock_guard<std::recursive_mutex> lock(ctx->devices_mutex);

    auto it = ctx->devices.find(index);
    if (it != ctx->devices.end()) {
        return it->second;
    }
    
    return NULL;
}

bool ggml_backend_numa_enable(ggml_backend_reg_t reg, ggml_numa_strategy numa_strategy) {
    if (numa_strategy == GGML_NUMA_STRATEGY_DISABLED) {
        GGML_LOG_WARN("%s: NUMA backend is disabled\n", __func__);
        return false;
    }

    auto ctx = (ggml_backend_numa_reg_context *)reg->context;

    std::lock_guard<std::recursive_mutex> lock(ctx->devices_mutex);

#ifndef __gnu_linux__
    GGML_LOG_WARN("%s: NUMA backend is only supported on Linux\n", __func__);
    return false;
#else
    // Enumerate NUMA nodes and create devices for each one
    size_t node_index = 0;
    while (true) {
        std::string path = "/sys/devices/system/node/node" + std::to_string(node_index);
        if (!std::filesystem::exists(path)) { 
            break; 
        }

        GGML_LOG_INFO("%s: found NUMA node %zu\n", __func__, node_index);

        // Create device for this NUMA node (our NUMA CPU devices start at 1)
        auto cpu_device = (ggml_backend_dev_t) ctx->backend_reg_cpu->iface.get_device(ctx->backend_reg_cpu, node_index + 1);
        if (cpu_device == NULL) {
            break;
        }
        
        ggml_backend_numa_backend_dev_context * device_ctx;
        ggml_backend_device * numa_device;

        if (!ctx->devices_initialized) {
            // Allocate a new NUMA device 
            device_ctx = new ggml_backend_numa_backend_dev_context();
            device_ctx->backend_dev_cpu = cpu_device;
            device_ctx->name            = "NUMA" + std::to_string(node_index);
            device_ctx->description     = "NUMA device " + std::to_string(node_index);

            if (device_ctx == NULL) {
                GGML_LOG_ERROR("%s: Failed to create NUMA backend device context\n", __func__);
                return false;
            }

            numa_device = new ggml_backend_device {
                /* .iface   = */ ggml_backend_numa_device_i,
                /* .reg     = */ reg,
                /* .context = */ device_ctx,
            };
            if (numa_device == NULL) {
                GGML_LOG_ERROR("%s: Failed to create NUMA backend device\n", __func__);
                delete device_ctx;
                return false;
            }

            ctx->devices[node_index] = numa_device;

        } else {
            // NUMA node devices already allocated
            numa_device = ctx->devices[node_index];
            device_ctx = (ggml_backend_numa_backend_dev_context *)numa_device->context;
        }

        auto cpu_backend = (ggml_backend_t) cpu_device->iface.init_backend(cpu_device, NULL);

        auto set_threads_fn = (decltype(ggml_backend_cpu_set_n_threads) *) ggml_backend_reg_get_proc_address(ctx->backend_reg_cpu, "ggml_backend_set_n_threads");
        if (!set_threads_fn) {
            GGML_LOG_ERROR("%s: Failed to get CPU backend set_n_threads function\n", __func__);
            return false;
        }

        auto set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(ctx->backend_reg_cpu, "ggml_backend_cpu_set_threadpool");
        if (!set_threadpool_fn) {
            GGML_LOG_ERROR("%s: Failed to get CPU backend set_threadpool function\n", __func__);
            return false;
        }

        auto set_numa_node_fn = (decltype(ggml_backend_cpu_set_numa_node) *) ggml_backend_reg_get_proc_address(ctx->backend_reg_cpu, "ggml_backend_cpu_set_numa_node");
        if (!set_numa_node_fn) {
            GGML_LOG_ERROR("%s: Failed to get CPU backend set_numa_node function\n", __func__);
            return false;
        }

        auto ggml_threadpool_new_fn = (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(ctx->backend_reg_cpu, "ggml_threadpool_new");
        if (!ggml_threadpool_new_fn) {
            GGML_LOG_ERROR("%s: Failed to get CPU backend threadpool_new function\n", __func__);
            return false;
        }

        auto ggml_threadpool_free_fn = (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(ctx->backend_reg_cpu, "ggml_threadpool_free");
        if (!ggml_threadpool_free_fn) {
            GGML_LOG_ERROR("%s: Failed to get CPU backend threadpool_free function\n", __func__);
            return false;
        }

        // Construct a threadpool for this NUMA node using only physical cores
        uint32_t cpu_count = 0;
        bool cpumask[GGML_MAX_N_THREADS];

        std::memset(cpumask, 0, sizeof(cpumask));

        // Enumerate CPUs and enable only physical cores (thread_siblings_list)
        for (uint32_t i = 0; i < GGML_MAX_N_THREADS; ++i) {
            std::string cpu_path = "/sys/devices/system/node/node" + std::to_string(node_index) + "/cpu" + std::to_string(i);
            if (std::filesystem::exists(cpu_path)) {
                if (numa_strategy == GGML_NUMA_STRATEGY_PHYSICAL) {
                    // Check if the CPU is a physical core
                    std::string topo_path = "/sys/devices/system/cpu/cpu" + std::to_string(i) + "/topology/thread_siblings_list";
                    std::ifstream cpu_file(topo_path);
                    if (cpu_file) {
                        std::string line;
                        if (std::getline(cpu_file, line)) {
                            // Only enable the first CPU in the siblings list (physical core)
                            size_t comma = line.find(',');
                            int phys_core;
                            
                            if (comma != std::string::npos) {
                                phys_core = std::stoi(line.substr(0, comma));
                            } else {
                                phys_core = std::stoi(line);
                            }
                            if ((int)i == phys_core) {
                                GGML_LOG_INFO("%s: enabling physical CPU %u for NUMA node %zu\n", __func__, i, node_index);

                                cpumask[i] = true;
                                cpu_count++;
                            }
                        }
                    }
                } else {
                    GGML_LOG_INFO("%s: enabling CPU %u for NUMA node %zu\n", __func__, i, node_index);

                    cpumask[i] = true;
                    cpu_count++;
                }
            }
        }

        device_ctx->numa_node = node_index;
        device_ctx->cpu_count = cpu_count;

        struct ggml_threadpool_params threadpool_params;

        ggml_threadpool_params_init(&threadpool_params, cpu_count);

        threadpool_params.paused = true;
        threadpool_params.strict_cpu = true;
        std::memcpy(&threadpool_params.cpumask, &cpumask, GGML_MAX_N_THREADS);

        auto threadpool = ggml_threadpool_new_fn(&threadpool_params);
        if (threadpool == NULL) {
            GGML_LOG_ERROR("%s: Failed to create threadpool for NUMA node %zu\n", __func__, node_index);
            return false;
        }
        
        // Enable threadpool switching
        set_numa_node_fn(cpu_backend, -1);

        // Set the number of threads for this NUMA node
        set_threads_fn(cpu_backend, cpu_count);

        // Set the threadpool for this NUMA node
        set_threadpool_fn(cpu_backend, threadpool);

        // Set the NUMA node for this CPU backend, locking the threadpool to this NUMA node
        set_numa_node_fn(cpu_backend, node_index);

        // Clean up the threadpool if it was already set
        if (device_ctx->threadpool != NULL) {
            ggml_threadpool_free_fn(device_ctx->threadpool);
            device_ctx->threadpool = threadpool;
        }

        // Start the worker thread for this NUMA node
        if (device_ctx->async.worker.joinable() == false) {
            device_ctx->async.worker = std::thread(numa_worker_loop, numa_device);
        }

        node_index++;
    }

    if (ctx->devices.empty()) {
        GGML_LOG_WARN("%s: No NUMA nodes found, NUMA backend not enabled\n", __func__);
        return false;
    }
    
    ctx->devices_initialized = true;

    GGML_LOG_INFO("%s: NUMA backend enabled with %zu nodes (strategy: %d)\n", __func__, ctx->devices.size(), numa_strategy);
    return true;
#endif
}

static void * ggml_backend_numa_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_numa_enable") == 0) {
        return (void *)ggml_backend_numa_enable;
    }

    return NULL;

    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_numa_reg_i = {
    /* .get_name         = */ ggml_backend_numa_reg_get_name,
    /* .get_device_count = */ ggml_backend_numa_reg_get_device_count,
    /* .get_device       = */ ggml_backend_numa_reg_get_device,
    /* .get_proc_address = */ ggml_backend_numa_get_proc_address,
};

ggml_backend_reg_t ggml_backend_numa_reg(ggml_backend_reg_t reg_cpu) {
    if (reg_cpu == NULL) {
        GGML_LOG_ERROR("%s: CPU backend is NULL\n", __func__);
        return NULL;
    }

    static ggml_backend_numa_reg_context ctx = {
        /* .backend_reg_cpu = */ reg_cpu,
        /* .devices_mutex   = */ {},
        /* .devices         = */ {},
    };

    static struct ggml_backend_reg ggml_backend_numa_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_numa_reg_i,
        /* .context     = */ &ctx,
    };

    // Enable NUMA backend with a default strategy
    // This can be overridden by the user at a later time by simply calling ggml_backend_numa_enable again
    ggml_backend_numa_enable(&ggml_backend_numa_reg, GGML_NUMA_STRATEGY_PHYSICAL);

    return &ggml_backend_numa_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_numa_reg)