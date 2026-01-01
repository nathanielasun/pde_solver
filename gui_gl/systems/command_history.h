#ifndef COMMAND_HISTORY_H
#define COMMAND_HISTORY_H

#include <memory>
#include <vector>
#include <string>
#include <functional>

// Base class for undoable commands (Command pattern)
class UICommand {
 public:
  virtual ~UICommand() = default;
  
  // Execute the command
  virtual void Execute() = 0;
  
  // Undo the command
  virtual void Undo() = 0;
  
  // Get description of the command
  virtual std::string Description() const = 0;
  
  // Check if command can be merged with another command
  virtual bool CanMergeWith(const UICommand* other) const { return false; }
  
  // Merge this command with another (for batching similar operations)
  virtual void Merge(const UICommand* other) {}
};

// Command history manager for undo/redo functionality
class CommandHistory {
 public:
  CommandHistory(size_t max_history = 50);
  
  // Execute a command and add it to history
  void Execute(std::unique_ptr<UICommand> cmd);
  
  // Undo the last command
  void Undo();
  
  // Redo the last undone command
  void Redo();
  
  // Check if undo is available
  bool CanUndo() const;
  
  // Check if redo is available
  bool CanRedo() const;
  
  // Get description of next undo operation
  std::string UndoDescription() const;
  
  // Get description of next redo operation
  std::string RedoDescription() const;
  
  // Clear all history
  void Clear();
  
  // Get history size
  size_t GetHistorySize() const { return history_.size(); }
  
  // Get current position in history
  size_t GetCurrentIndex() const { return current_index_; }
  
  // Set maximum history depth
  void SetMaxHistory(size_t max_history);
  
  // Get maximum history depth
  size_t GetMaxHistory() const { return max_history_; }

 private:
  std::vector<std::unique_ptr<UICommand>> history_;
  size_t current_index_ = 0;
  size_t max_history_ = 50;
  
  // Trim history if it exceeds max_history_
  void TrimHistory();
};

// Concrete command implementations

// Command for changing a string value
class SetStringCommand : public UICommand {
 public:
  SetStringCommand(std::string* target, const std::string& new_value, const std::string& description);
  
  void Execute() override;
  void Undo() override;
  std::string Description() const override;
  bool CanMergeWith(const UICommand* other) const override;
  void Merge(const UICommand* other) override;

 private:
  std::string* target_;
  std::string new_value_;
  std::string old_value_;
  std::string description_;
};

// Command for changing a double value
class SetDoubleCommand : public UICommand {
 public:
  SetDoubleCommand(double* target, double new_value, const std::string& description);
  
  void Execute() override;
  void Undo() override;
  std::string Description() const override;

 private:
  double* target_;
  double new_value_;
  double old_value_;
  std::string description_;
};

// Command for changing an int value
class SetIntCommand : public UICommand {
 public:
  SetIntCommand(int* target, int new_value, const std::string& description);
  
  void Execute() override;
  void Undo() override;
  std::string Description() const override;

 private:
  int* target_;
  int new_value_;
  int old_value_;
  std::string description_;
};

// Command for changing a bool value
class SetBoolCommand : public UICommand {
 public:
  SetBoolCommand(bool* target, bool new_value, const std::string& description);
  
  void Execute() override;
  void Undo() override;
  std::string Description() const override;

 private:
  bool* target_;
  bool new_value_;
  bool old_value_;
  std::string description_;
};

// Macro command for grouping multiple commands
class MacroCommand : public UICommand {
 public:
  MacroCommand(const std::string& description);
  
  void AddCommand(std::unique_ptr<UICommand> cmd);
  void Execute() override;
  void Undo() override;
  std::string Description() const override;

 private:
  std::vector<std::unique_ptr<UICommand>> commands_;
  std::string description_;
};

#endif  // COMMAND_HISTORY_H

