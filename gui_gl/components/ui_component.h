#ifndef UI_COMPONENT_H
#define UI_COMPONENT_H

#include <string>
#include <memory>
#include <functional>
#include <vector>
#include <map>

// Base class for all UI components
class UIComponent {
 public:
  virtual ~UIComponent() = default;
  
  // Render the component
  virtual void Render() = 0;
  
  // Get component name for identification
  virtual std::string GetName() const = 0;
  
  // Check if component should be visible
  virtual bool IsVisible() const { return true; }
  
  // Set visibility
  virtual void SetVisible(bool visible) { visible_ = visible; }
  
  // Get visibility
  bool GetVisible() const { return visible_; }

 protected:
  bool visible_ = true;
};

// Component manager for organizing and rendering components
class ComponentManager {
 public:
  // Register a component
  void RegisterComponent(std::unique_ptr<UIComponent> component);
  
  // Render all visible components
  void RenderAll();
  
  // Get component by name
  UIComponent* GetComponent(const std::string& name);
  
  // Get all component names
  std::vector<std::string> GetComponentNames() const;
  
  // Remove component
  void RemoveComponent(const std::string& name);
  
  // Clear all components
  void Clear();

 private:
  std::map<std::string, std::unique_ptr<UIComponent>> components_;
};

#endif  // UI_COMPONENT_H

