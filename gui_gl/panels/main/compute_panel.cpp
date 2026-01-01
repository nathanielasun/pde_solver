#include "compute_panel.h"

#include "app_helpers.h"
#include "systems/backend_capabilities.h"
#include "systems/solver_method_registry.h"
#include "styles/ui_style.h"
#include "ui_helpers.h"
#include "imgui.h"

#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

namespace {

std::vector<std::string> DefaultComponents() {
  return {"backend", "solver_method", "solver_options"};
}

BackendKind BackendKindFromName(const std::string& name) {
  if (name == "CPU") {
    return BackendKind::CPU;
  }
  if (name == "CUDA") {
    return BackendKind::CUDA;
  }
  if (name == "Metal") {
    return BackendKind::Metal;
  }
  if (name == "TPU") {
    return BackendKind::TPU;
  }
  return BackendKind::Auto;
}

std::string BackendLabelFromKind(BackendKind kind) {
  switch (kind) {
    case BackendKind::CPU:
      return "CPU";
    case BackendKind::CUDA:
      return "CUDA";
    case BackendKind::Metal:
      return "Metal";
    case BackendKind::TPU:
      return "TPU";
    default:
      return "Auto";
  }
}

struct BackendChoice {
  BackendKind kind;
  std::string label;
};

std::vector<BackendChoice> BuildBackendChoices(BackendUIRegistry* registry,
                                               BackendKind current_backend) {
  std::vector<BackendChoice> choices;
  choices.push_back({BackendKind::Auto, "Auto"});

  if (registry) {
    for (const auto& name : registry->GetAvailableBackends()) {
      BackendKind kind = BackendKindFromName(name);
      if (kind == BackendKind::Auto) {
        continue;
      }
      choices.push_back({kind, name});
    }
  } else {
    const char* fallback_names[] = {"CPU", "CUDA", "Metal", "TPU"};
    for (const char* name : fallback_names) {
      choices.push_back({BackendKindFromName(name), name});
    }
  }

  bool current_present = false;
  for (const auto& choice : choices) {
    if (choice.kind == current_backend) {
      current_present = true;
      break;
    }
  }
  if (!current_present && current_backend != BackendKind::Auto) {
    std::string label = BackendLabelFromKind(current_backend);
    label += " (Unavailable)";
    choices.push_back({current_backend, label});
  }

  return choices;
}

void RenderBackendSection(ComputePanelState& state) {
  ImGui::Text("Backend");

  BackendKind current_backend = BackendFromIndex(state.backend_index);
  auto choices = BuildBackendChoices(state.backend_registry, current_backend);
  std::vector<const char*> labels;
  labels.reserve(choices.size());
  for (const auto& choice : choices) {
    labels.push_back(choice.label.c_str());
  }

  int current_index = 0;
  for (size_t i = 0; i < choices.size(); ++i) {
    if (choices[i].kind == current_backend) {
      current_index = static_cast<int>(i);
      break;
    }
  }

  ImGui::SetNextItemWidth(state.input_width);
  if (UIInput::Combo("##backend_selector", &current_index, labels.data(),
                     static_cast<int>(labels.size()))) {
    BackendKind selected = choices[current_index].kind;
    state.backend_index = BackendToIndex(selected);
    state.prefs_changed = true;
  }

  if (current_backend == BackendKind::Auto) {
    ImGui::TextDisabled("Auto selects the best available backend at solve time.");
  }
}

void RenderSolverMethodSection(ComputePanelState& state,
                               BackendUIProvider* backend_provider) {
  BackendCapabilities caps = backend_provider ? backend_provider->GetCapabilities()
                                              : BackendCapabilities{};
  auto& registry = SolverMethodRegistry::Instance();
  auto methods = registry.GetMethods();

  std::vector<SolveMethod> available_methods;
  std::vector<std::string> labels_storage;
  std::vector<const char*> labels;

  if (methods.empty()) {
    const SolveMethod fallback_methods[] = {
      SolveMethod::Jacobi,
      SolveMethod::GaussSeidel,
      SolveMethod::SOR,
      SolveMethod::CG,
      SolveMethod::BiCGStab,
      SolveMethod::GMRES,
      SolveMethod::MultigridVcycle
    };
    const char* fallback_labels[] = {
      "Jacobi",
      "Gauss-Seidel",
      "SOR",
      "CG",
      "BiCGStab",
      "GMRES",
      "Multigrid V-cycle"
    };
    for (size_t i = 0; i < std::size(fallback_methods); ++i) {
      available_methods.push_back(fallback_methods[i]);
      labels_storage.push_back(fallback_labels[i]);
    }
  } else {
    for (SolveMethod method : methods) {
      if (!caps.supported_methods.empty() && !caps.SupportsMethod(method)) {
        continue;
      }
      const SolverMethodMetadata* metadata = registry.GetMetadata(method);
      if (metadata) {
        labels_storage.push_back(metadata->name);
      } else {
        labels_storage.push_back("Unknown method");
      }
      available_methods.push_back(method);
    }
  }

  if (available_methods.empty()) {
    ImGui::TextDisabled("No solver methods available for this backend.");
    return;
  }

  labels.reserve(labels_storage.size());
  for (const auto& label : labels_storage) {
    labels.push_back(label.c_str());
  }

  SolveMethod current_method = MethodFromIndex(state.method_index);
  int current_index = 0;
  for (size_t i = 0; i < available_methods.size(); ++i) {
    if (available_methods[i] == current_method) {
      current_index = static_cast<int>(i);
      break;
    }
  }

  ImGui::Text("Solver Method");
  ImGui::SetNextItemWidth(state.input_width);
  if (UIInput::Combo("##solver_method", &current_index, labels.data(),
                     static_cast<int>(labels.size()))) {
    SolveMethod selected = available_methods[static_cast<size_t>(current_index)];
    int index = MethodToIndex(selected);
    state.method_index = index;
    state.pref_method_index = index;
    state.prefs_changed = true;
  }

  const SolverMethodMetadata* metadata = registry.GetMetadata(current_method);
  if (metadata && !metadata->description.empty()) {
    ImGui::TextDisabled("%s", metadata->description.c_str());
  }
}

void RenderSolverOptionsSection(ComputePanelState& state,
                                BackendKind current_backend,
                                BackendUIProvider* backend_provider) {
  ImGui::Text("Options");

  ImGui::SetNextItemWidth(state.input_width);
  int threads = state.thread_count;
  if (UIInput::InputInt("Threads", &threads, 1, 1)) {
    threads = std::max(1, std::min(threads, state.max_threads));
    state.thread_count = threads;
    state.prefs_changed = true;
  }

  if (ImGui::CollapsingHeader("Advanced Options")) {
    SolveMethod method = MethodFromIndex(state.method_index);
    if (method == SolveMethod::SOR) {
      ImGui::SetNextItemWidth(state.input_width);
      double omega = state.sor_omega;
      if (UIInput::InputDouble("SOR omega", &omega, 0.01, 0.1, "%.3f")) {
        omega = std::max(0.1, std::min(omega, 1.99));
        state.sor_omega = omega;
        state.pref_sor_omega = omega;
        state.prefs_changed = true;
      }
    }

    if (method == SolveMethod::GMRES) {
      ImGui::SetNextItemWidth(state.input_width);
      int restart = state.gmres_restart;
      if (UIInput::InputInt("GMRES restart", &restart, 1, 1)) {
        restart = std::max(2, std::min(restart, 4096));
        state.gmres_restart = restart;
        state.pref_gmres_restart = restart;
        state.prefs_changed = true;
      }
    }

    if (current_backend == BackendKind::Metal) {
      ImGui::Separator();
      ImGui::Text("Metal tuning");

      ImGui::SetNextItemWidth(state.input_width);
      int reduce = state.metal_reduce_interval;
      if (UIInput::InputInt("Reduce interval", &reduce, 1, 1)) {
        reduce = std::max(1, std::min(reduce, 1024));
        state.metal_reduce_interval = reduce;
        state.pref_metal_reduce_interval = reduce;
        state.prefs_changed = true;
      }

      ImGui::SetNextItemWidth(state.input_width);
      int tgx = state.metal_tg_x;
      if (UIInput::InputInt("Threadgroup X", &tgx, 1, 1)) {
        tgx = std::max(1, std::min(tgx, 1024));
        state.metal_tg_x = tgx;
        state.pref_metal_tg_x = tgx;
        state.prefs_changed = true;
      }

      ImGui::SetNextItemWidth(state.input_width);
      int tgy = state.metal_tg_y;
      if (UIInput::InputInt("Threadgroup Y", &tgy, 1, 1)) {
        tgy = std::max(1, std::min(tgy, 1024));
        state.metal_tg_y = tgy;
        state.pref_metal_tg_y = tgy;
        state.prefs_changed = true;
      }

      if (ImGui::TreeNode("Metal defaults")) {
        ImGui::SetNextItemWidth(state.input_width);
        int pref_reduce = state.pref_metal_reduce_interval;
        if (UIInput::InputInt("Default reduce interval", &pref_reduce, 1, 1)) {
          pref_reduce = std::max(1, std::min(pref_reduce, 1024));
          state.pref_metal_reduce_interval = pref_reduce;
          state.prefs_changed = true;
        }

        int pref_tg[2] = {state.pref_metal_tg_x, state.pref_metal_tg_y};
        if (UIInput::InputInt2("Default threadgroup (x,y)", pref_tg)) {
          state.pref_metal_tg_x = std::max(0, pref_tg[0]);
          state.pref_metal_tg_y = std::max(0, pref_tg[1]);
          state.prefs_changed = true;
        }
        ImGui::TextDisabled("0 = auto");

        if (UIButton::Button("Apply defaults to current tuning", UIButton::Size::Small,
                             UIButton::Variant::Secondary)) {
          state.metal_reduce_interval = state.pref_metal_reduce_interval;
          state.metal_tg_x = state.pref_metal_tg_x;
          state.metal_tg_y = state.pref_metal_tg_y;
        }
        ImGui::SameLine();
        if (UIButton::Button("Reset defaults", UIButton::Size::Small, UIButton::Variant::Secondary)) {
          state.pref_metal_reduce_interval = 10;
          state.pref_metal_tg_x = 0;
          state.pref_metal_tg_y = 0;
          state.prefs_changed = true;
        }
        ImGui::TreePop();
      }
    }

    if (backend_provider) {
      ImGui::Separator();
      ImGui::Text("Backend details");
      backend_provider->RenderOptionsUI();
    }
  }
}

}  // namespace

void RenderComputePanel(ComputePanelState& state, const std::vector<std::string>& components) {
  if (state.initialize_backend_registry) {
    state.initialize_backend_registry();
  }

  BackendKind current_backend = BackendFromIndex(state.backend_index);
  BackendUIProvider* backend_provider = nullptr;
  if (state.backend_registry) {
    backend_provider = state.backend_registry->GetProvider(current_backend);
    if (!backend_provider) {
      backend_provider = state.backend_registry->GetProvider(BackendKind::CPU);
    }
  }

  const std::vector<std::string> default_components = DefaultComponents();
  const std::vector<std::string>& component_list =
      components.empty() ? default_components : components;

  for (size_t i = 0; i < component_list.size(); ++i) {
    const std::string& id = component_list[i];
    if (id == "backend") {
      RenderBackendSection(state);
    } else if (id == "solver_method") {
      RenderSolverMethodSection(state, backend_provider);
    } else if (id == "solver_options") {
      RenderSolverOptionsSection(state, current_backend, backend_provider);
    } else {
      DrawUnknownComponentPlaceholder(id.c_str());
    }

    if (i + 1 < component_list.size()) {
      ImGui::Spacing();
    }
  }
}
