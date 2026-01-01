#ifndef DATASET_TOOLS_H
#define DATASET_TOOLS_H

#include <filesystem>
#include <string>

struct DatasetIndexResult {
  std::string json;
  int runs_total = 0;
  int runs_completed = 0;
  int runs_time_series = 0;
  int runs_steady = 0;
  int total_frames = 0;
  int missing_outputs = 0;
  int monitor_warning_runs = 0;
};

struct DatasetCleanupResult {
  int removed_summaries = 0;
  int removed_metadata = 0;
  int removed_empty_dirs = 0;
  int skipped = 0;
};

bool BuildDatasetIndex(const std::filesystem::path& root,
                       DatasetIndexResult* out,
                       std::string* error);
bool WriteDatasetIndex(const std::filesystem::path& path,
                       const DatasetIndexResult& result,
                       std::string* error);
bool CleanupDataset(const std::filesystem::path& root,
                    bool dry_run,
                    bool remove_empty_dirs,
                    DatasetCleanupResult* out,
                    std::string* error);

#endif  // DATASET_TOOLS_H
