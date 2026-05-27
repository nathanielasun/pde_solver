#ifndef LATEX_RENDER_SERVICE_H
#define LATEX_RENDER_SERVICE_H

#include "latex/latex_texture.h"
#include "latex/latex_bitmap.h"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>

class LatexRenderService {
 public:
  static LatexRenderService& Instance();

  void Init(const std::string& microtex_res_root = "");
  void Shutdown();

  void InvalidateAll();
  void SetGlobalStyle(const LatexRenderStyle& style);

  // Marks texture dirty and queues render after debounce (non-blocking).
  void RequestRender(LatexTexture& tex, const std::string& source, const LatexRenderStyle& style);

  // Call each frame on the main thread: uploads completed bitmaps to GL textures.
  void PollCompletedRenders();

  static constexpr int kDebounceMs = 250;

 private:
  struct CacheKey {
    std::string source;
    std::string color;
    int font_size = 0;
    bool operator==(const CacheKey& o) const;
  };
  struct CacheKeyHash {
    size_t operator()(const CacheKey& k) const;
  };

  struct Job {
    LatexTexture* tex = nullptr;
    std::string source;
    LatexRenderStyle style;
    size_t generation = 0;
  };

  struct Completed {
    LatexTexture* tex = nullptr;
    LatexBitmap bitmap;
    std::string source;
    std::string color;
    int font_size = 0;
    size_t generation = 0;
  };

  LatexRenderService() = default;
  void WorkerLoop();
  bool TryGetCached(const CacheKey& key, LatexBitmap* out);
  void StoreCache(const CacheKey& key, const LatexBitmap& bitmap);

  std::thread worker_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool running_ = false;
  bool stop_ = false;
  std::queue<Job> jobs_;
  std::vector<Completed> completed_;
  std::unordered_map<CacheKey, LatexBitmap, CacheKeyHash> cache_;
  LatexRenderStyle global_style_;
};

// Thin compatibility wrapper used by panels.
void UpdateLatexTexture(LatexTexture& tex, const std::string& source, const std::string& color,
                        int font_size);

#endif  // LATEX_RENDER_SERVICE_H
