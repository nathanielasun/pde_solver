#include "solver.h"
#include "vtk_io.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {
bool GridEqual(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  if (a.empty()) {
    return true;
  }
  return std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}
}

int main() {
  SolveInput input;
  input.domain = Domain{0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 8, 8, 1};
  input.pde = {};
  input.pde.a = 1.0;
  input.pde.b = 1.0;
  input.pde.ut = 1.0;
  input.pde.f = 0.0;
  input.bc = {};
  input.time.enabled = true;
  input.time.t_start = 0.0;
  input.time.dt = 0.1;
  input.time.frames = 4;

  std::vector<std::vector<double>> frames;
  const int checkpoint_frame = 1;
  const std::filesystem::path temp_path =
      std::filesystem::temp_directory_path() /
      ("pde_checkpoint_" + GenerateRandomTag(8) + ".txt");

  auto capture_cb = [&](int frame, double t, const std::vector<double>& grid,
                        const std::vector<double>* velocity) -> bool {
    if (frame >= 0 && frame < input.time.frames) {
      frames.push_back(grid);
    }
    if (frame == checkpoint_frame) {
      CheckpointData checkpoint;
      checkpoint.domain = input.domain;
      checkpoint.grid = grid;
      if (velocity && !velocity->empty()) {
        checkpoint.velocity = *velocity;
      }
      checkpoint.t_current = t;
      checkpoint.frame_index = frame;
      checkpoint.pde = input.pde;
      checkpoint.bc = input.bc;
      VtkWriteResult write_result = WriteCheckpoint(temp_path.string(), checkpoint);
      if (!write_result.ok) {
        std::cerr << "checkpoint write failed: " << write_result.error << "\n";
        return false;
      }
    }
    return true;
  };

  SolveOutput base = SolvePDETimeSeries(input, capture_cb, ProgressCallback());
  if (!base.error.empty()) {
    std::cerr << "base solve failed: " << base.error << "\n";
    return 1;
  }
  if (static_cast<int>(frames.size()) != input.time.frames) {
    std::cerr << "unexpected frame count in base solve\n";
    return 1;
  }

  CheckpointData checkpoint;
  VtkReadResult read_result = ReadCheckpoint(temp_path.string(), &checkpoint);
  if (!read_result.ok) {
    std::cerr << "checkpoint read failed: " << read_result.error << "\n";
    return 1;
  }
  if (!GridEqual(checkpoint.grid, frames[checkpoint_frame])) {
    std::cerr << "checkpoint grid mismatch after read\n";
    return 1;
  }

  SolveInput restart_input = input;
  restart_input.time.t_start = checkpoint.t_current;
  restart_input.time.frames = std::max(1, input.time.frames - checkpoint.frame_index);
  restart_input.initial_grid = checkpoint.grid;
  restart_input.initial_velocity = checkpoint.velocity;

  std::vector<std::vector<double>> restart_frames;
  auto restart_cb = [&](int frame, double, const std::vector<double>& grid,
                        const std::vector<double>*) -> bool {
    restart_frames.push_back(grid);
    return true;
  };

  SolveOutput restarted = SolvePDETimeSeries(restart_input, restart_cb, ProgressCallback());
  if (!restarted.error.empty()) {
    std::cerr << "restart solve failed: " << restarted.error << "\n";
    return 1;
  }

  if (restart_frames.empty()) {
    std::cerr << "restart produced no frames\n";
    return 1;
  }
  if (static_cast<int>(restart_frames.size()) != restart_input.time.frames) {
    std::cerr << "unexpected restart frame count\n";
    return 1;
  }

  for (size_t i = 0; i < restart_frames.size(); ++i) {
    const size_t base_index = static_cast<size_t>(checkpoint_frame) + i;
    if (base_index >= frames.size()) {
      std::cerr << "restart frame index out of range\n";
      return 1;
    }
    if (!GridEqual(restart_frames[i], frames[base_index])) {
      std::cerr << "restart grid mismatch at frame " << base_index << "\n";
      return 1;
    }
  }

  std::error_code ec;
  std::filesystem::remove(temp_path, ec);
  std::cout << "checkpoint equivalence test passed\n";
  return 0;
}
