#include "command_history.h"
#include <algorithm>
#include <cassert>

// CommandHistory implementation
CommandHistory::CommandHistory(size_t max_history) : max_history_(max_history) {}

void CommandHistory::Execute(std::unique_ptr<UICommand> cmd) {
  // Remove any commands after current_index_ (when we're in the middle of history)
  if (current_index_ < history_.size()) {
    history_.erase(history_.begin() + current_index_, history_.end());
  }
  
  // Try to merge with last command if possible
  if (!history_.empty() && history_.back()->CanMergeWith(cmd.get())) {
    history_.back()->Merge(cmd.get());
    history_.back()->Execute();
  } else {
    // Add new command
    cmd->Execute();
    history_.push_back(std::move(cmd));
    current_index_ = history_.size();
  }
  
  TrimHistory();
}

void CommandHistory::Undo() {
  if (!CanUndo()) {
    return;
  }
  
  current_index_--;
  history_[current_index_]->Undo();
}

void CommandHistory::Redo() {
  if (!CanRedo()) {
    return;
  }
  
  history_[current_index_]->Execute();
  current_index_++;
}

bool CommandHistory::CanUndo() const {
  return current_index_ > 0;
}

bool CommandHistory::CanRedo() const {
  return current_index_ < history_.size();
}

std::string CommandHistory::UndoDescription() const {
  if (!CanUndo()) {
    return "";
  }
  return history_[current_index_ - 1]->Description();
}

std::string CommandHistory::RedoDescription() const {
  if (!CanRedo()) {
    return "";
  }
  return history_[current_index_]->Description();
}

void CommandHistory::Clear() {
  history_.clear();
  current_index_ = 0;
}

void CommandHistory::SetMaxHistory(size_t max_history) {
  max_history_ = max_history;
  TrimHistory();
}

void CommandHistory::TrimHistory() {
  if (history_.size() <= max_history_) {
    return;
  }
  
  // Remove oldest commands, keeping current_index_ valid
  const size_t to_remove = history_.size() - max_history_;
  if (current_index_ > to_remove) {
    history_.erase(history_.begin(), history_.begin() + to_remove);
    current_index_ -= to_remove;
  } else {
    // If we're removing past current_index_, just clear everything
    history_.clear();
    current_index_ = 0;
  }
}

// SetStringCommand implementation
SetStringCommand::SetStringCommand(std::string* target, const std::string& new_value, const std::string& description)
  : target_(target), new_value_(new_value), old_value_(target ? *target : ""), description_(description) {}

void SetStringCommand::Execute() {
  if (target_) {
    old_value_ = *target_;
    *target_ = new_value_;
  }
}

void SetStringCommand::Undo() {
  if (target_) {
    *target_ = old_value_;
  }
}

std::string SetStringCommand::Description() const {
  return description_;
}

bool SetStringCommand::CanMergeWith(const UICommand* other) const {
  const SetStringCommand* cmd = dynamic_cast<const SetStringCommand*>(other);
  return cmd && cmd->target_ == target_ && description_ == cmd->description_;
}

void SetStringCommand::Merge(const UICommand* other) {
  const SetStringCommand* cmd = dynamic_cast<const SetStringCommand*>(other);
  if (cmd) {
    new_value_ = cmd->new_value_;
  }
}

// SetDoubleCommand implementation
SetDoubleCommand::SetDoubleCommand(double* target, double new_value, const std::string& description)
  : target_(target), new_value_(new_value), old_value_(target ? *target : 0.0), description_(description) {}

void SetDoubleCommand::Execute() {
  if (target_) {
    old_value_ = *target_;
    *target_ = new_value_;
  }
}

void SetDoubleCommand::Undo() {
  if (target_) {
    *target_ = old_value_;
  }
}

std::string SetDoubleCommand::Description() const {
  return description_;
}

// SetIntCommand implementation
SetIntCommand::SetIntCommand(int* target, int new_value, const std::string& description)
  : target_(target), new_value_(new_value), old_value_(target ? *target : 0), description_(description) {}

void SetIntCommand::Execute() {
  if (target_) {
    old_value_ = *target_;
    *target_ = new_value_;
  }
}

void SetIntCommand::Undo() {
  if (target_) {
    *target_ = old_value_;
  }
}

std::string SetIntCommand::Description() const {
  return description_;
}

// SetBoolCommand implementation
SetBoolCommand::SetBoolCommand(bool* target, bool new_value, const std::string& description)
  : target_(target), new_value_(new_value), old_value_(target ? *target : false), description_(description) {}

void SetBoolCommand::Execute() {
  if (target_) {
    old_value_ = *target_;
    *target_ = new_value_;
  }
}

void SetBoolCommand::Undo() {
  if (target_) {
    *target_ = old_value_;
  }
}

std::string SetBoolCommand::Description() const {
  return description_;
}

// MacroCommand implementation
MacroCommand::MacroCommand(const std::string& description) : description_(description) {}

void MacroCommand::AddCommand(std::unique_ptr<UICommand> cmd) {
  commands_.push_back(std::move(cmd));
}

void MacroCommand::Execute() {
  for (auto& cmd : commands_) {
    cmd->Execute();
  }
}

void MacroCommand::Undo() {
  // Undo in reverse order
  for (auto it = commands_.rbegin(); it != commands_.rend(); ++it) {
    (*it)->Undo();
  }
}

std::string MacroCommand::Description() const {
  return description_;
}

