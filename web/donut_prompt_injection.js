/**
 * DonutPromptInjection - Dynamic category/style dropdown handler
 *
 * This extension makes the style dropdown update dynamically based on the
 * selected category in the DonutPromptInjection node.
 */

import { app } from "../../scripts/app.js";

// Cache for styles by category (fetched once from server)
let stylesByCategory = null;

// Cache for user's previous selections per category
const userSelections = {};

/**
 * Fetch styles organized by category from the server
 */
async function fetchStylesByCategory() {
    if (stylesByCategory !== null) {
        return stylesByCategory;
    }

    try {
        const response = await fetch('/donut/styles/by_category');
        if (response.ok) {
            stylesByCategory = await response.json();
            return stylesByCategory;
        }
    } catch (error) {
        console.error('[DonutPromptInjection] Failed to fetch styles:', error);
    }
    return null;
}

/**
 * Update the style widget options based on selected category
 */
function updateStyleOptions(node, categoryWidget, styleWidget, newCategory) {
    if (!stylesByCategory || !stylesByCategory[newCategory]) {
        return;
    }

    // Store current selection for the old category
    const oldCategory = categoryWidget._lastCategory;
    if (oldCategory && styleWidget.value) {
        userSelections[oldCategory] = styleWidget.value;
    }

    // Get styles for the new category, add "Random" at the start
    const categoryStyles = stylesByCategory[newCategory];
    const newStyles = ["Random", ...categoryStyles];

    // Update the widget options
    styleWidget.options.values = newStyles;

    // Restore previous selection if user had one, otherwise keep current or use "None"
    if (userSelections[newCategory] && newStyles.includes(userSelections[newCategory])) {
        styleWidget.value = userSelections[newCategory];
    } else if (!newStyles.includes(styleWidget.value)) {
        // Current value not in new list, default to "None"
        styleWidget.value = "None";
    }

    // Remember current category
    categoryWidget._lastCategory = newCategory;

    // Trigger widget change callback if it exists
    if (styleWidget.callback) {
        styleWidget.callback(styleWidget.value);
    }
}

/**
 * Initialize the dynamic dropdown for a node
 */
async function initializeDynamicDropdown(node) {
    // Find the category and style widgets
    const categoryWidget = node.widgets?.find(w => w.name === 'category');
    const styleWidget = node.widgets?.find(w => w.name === 'style');

    if (!categoryWidget || !styleWidget) {
        return;
    }

    // Fetch styles if not already cached
    const styles = await fetchStylesByCategory();
    if (!styles) {
        console.warn('[DonutPromptInjection] Could not load styles');
        return;
    }

    // Store original callback
    const originalCallback = categoryWidget.callback;

    // Set up the category change handler
    categoryWidget.callback = function(value) {
        updateStyleOptions(node, categoryWidget, styleWidget, value);

        // Call original callback if it exists
        if (typeof originalCallback === 'function') {
            originalCallback.call(this, value);
        }
    };

    // Initialize with current category
    categoryWidget._lastCategory = categoryWidget.value;
    updateStyleOptions(node, categoryWidget, styleWidget, categoryWidget.value);
}

// Register the extension
app.registerExtension({
    name: "DonutNodes.PromptInjection",

    async nodeCreated(node) {
        // Check if this is our node
        if (node.comfyClass === "DonutPromptInjection") {
            // Small delay to ensure widgets are fully initialized
            setTimeout(() => {
                initializeDynamicDropdown(node);
            }, 100);
        }
    },

    async loadedGraphNode(node) {
        // Also handle nodes loaded from saved workflows
        if (node.comfyClass === "DonutPromptInjection") {
            setTimeout(() => {
                initializeDynamicDropdown(node);
            }, 100);
        }
    }
});
