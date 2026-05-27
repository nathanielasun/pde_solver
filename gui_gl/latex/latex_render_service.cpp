#include "latex/latex_render_service.h"

#include "latex/latex_preview_draw.h"

#ifdef USE_MICROTEX_LATEX_PREVIEW
#include "latex/microtex_rgba_graphic.h"
#endif

#include <utility>

bool LatexRenderService::CacheKey::operator==(const CacheKey& o) const {
  return source == o.source && color == o.color && font_size == o.font_size;
}

size_t LatexRenderService::CacheKeyHash::operator()(const CacheKey& k) const {
  std::hash<std::string> h;
  size_t v = h(k.source);
  v ^= h(k.color) + 0x9e3779b9 + (v << 6) + (v >> 2);
  v ^= static_cast<size_t>(k.font_size) + 0x9e3779b9 + (v << 6) + (v >> 2);
  return v;
}

LatexRenderService& LatexRenderService::Instance() {
  static LatexRenderService inst;
  return inst;
}

void LatexRenderService::Init(const std::string& microtex_res_root) {
#ifdef USE_MICROTEX_LATEX_PREVIEW
  if (!microtex_res_root.empty()) {
    MicroTeXInit(microtex_res_root);
  }
#else
  (void)microtex_res_root;
#endif
  std::lock_guard<std::mutex> lock(mutex_);
  if (running_) {
    return;
  }
  stop_ = false;
  running_ = true;
  worker_ = std::thread([this]() { WorkerLoop(); });
}

void LatexRenderService::Shutdown() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_) {
      return;
    }
    stop_ = true;
  }
  cv_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
#ifdef USE_MICROTEX_LATEX_PREVIEW
  MicroTeXShutdown();
#endif
  std::lock_guard<std::mutex> lock(mutex_);
  running_ = false;
  while (!jobs_.empty()) {
    jobs_.pop();
  }
  completed_.clear();
  cache_.clear();
}

void LatexRenderService::InvalidateAll() {
  std::lock_guard<std::mutex> lock(mutex_);
  cache_.clear();
}

void LatexRenderService::SetGlobalStyle(const LatexRenderStyle& style) {
  std::lock_guard<std::mutex> lock(mutex_);
  global_style_ = style;
  cache_.clear();
}

bool LatexRenderService::TryGetCached(const CacheKey& key, LatexBitmap* out) {
  const auto it = cache_.find(key);
  if (it == cache_.end()) {
    return false;
  }
  *out = it->second;
  return true;
}

void LatexRenderService::StoreCache(const CacheKey& key, const LatexBitmap& bitmap) {
  cache_[key] = bitmap;
}

void LatexRenderService::RequestRender(LatexTexture& tex, const std::string& source,
                                       const LatexRenderStyle& style) {
  const auto now = std::chrono::steady_clock::now();
  if (source != tex.source || style.fg_hex != tex.color || style.font_size != tex.font_size) {
    tex.source = source;
    tex.color = style.fg_hex;
    tex.font_size = style.font_size;
    tex.dirty = true;
    tex.pending = false;
    tex.last_edit = now;
  }
  if (!tex.dirty) {
    return;
  }
  if (now - tex.last_edit < std::chrono::milliseconds(kDebounceMs)) {
    return;
  }

  if (source.empty()) {
    tex.dirty = false;
    tex.pending = false;
    tex.error.clear();
    tex.last_rendered.clear();
    return;
  }

  if (source == tex.last_rendered && tex.texture != 0 && tex.error.empty()) {
    tex.dirty = false;
    return;
  }

  tex.pending = true;
  tex.dirty = false;

  Job job;
  job.tex = &tex;
  job.source = source;
  job.style = style;
  job.generation = reinterpret_cast<size_t>(&tex);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    jobs_.push(std::move(job));
  }
  cv_.notify_one();
}

void LatexRenderService::WorkerLoop() {
  while (true) {
    Job job;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this]() { return stop_ || !jobs_.empty(); });
      if (stop_ && jobs_.empty()) {
        return;
      }
      job = std::move(jobs_.front());
      jobs_.pop();
    }

    CacheKey key{job.source, job.style.fg_hex, job.style.font_size};
    LatexBitmap bitmap;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!TryGetCached(key, &bitmap)) {
        bitmap = RenderLatexToBitmap(job.source, job.style);
        if (bitmap.error.empty() && !bitmap.rgba.empty()) {
          StoreCache(key, bitmap);
        }
      }
    }

    Completed done;
    done.tex = job.tex;
    done.bitmap = std::move(bitmap);
    done.source = job.source;
    done.color = job.style.fg_hex;
    done.font_size = job.style.font_size;
    done.generation = job.generation;

    std::lock_guard<std::mutex> lock(mutex_);
    completed_.push_back(std::move(done));
  }
}

void LatexRenderService::PollCompletedRenders() {
  std::vector<Completed> batch;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    batch.swap(completed_);
  }

  for (auto& done : batch) {
    if (!done.tex) {
      continue;
    }
    LatexTexture& tex = *done.tex;
    if (done.source != tex.source || done.color != tex.color || done.font_size != tex.font_size) {
      tex.pending = false;
      continue;
    }

    tex.pending = false;
    tex.last_rendered = done.source;
    tex.error.clear();

    if (!done.bitmap.error.empty()) {
      tex.error = done.bitmap.error;
      continue;
    }
    if (done.bitmap.rgba.empty()) {
      tex.error = "render produced empty bitmap";
      continue;
    }

    std::string upload_error;
    if (!UploadTextureFromRGBA(&tex.texture, &tex.width, &tex.height, done.bitmap.rgba,
                               done.bitmap.width, done.bitmap.height, &upload_error)) {
      tex.error = upload_error.empty() ? "texture upload failed" : upload_error;
    }
  }
}

void UpdateLatexTexture(LatexTexture& tex, const std::string& source, const std::string& color,
                        int font_size) {
  LatexRenderStyle style;
  style.fg_hex = color;
  style.font_size = font_size;
  LatexRenderService::Instance().RequestRender(tex, source, style);
}
