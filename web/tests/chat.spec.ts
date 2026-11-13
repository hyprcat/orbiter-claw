import { test, expect } from "@playwright/test";

test.describe("Chat page", () => {
  test("loads chat page with expected elements", async ({ page }) => {
    await page.goto("/chat");

    // Page title
    await expect(page).toHaveTitle("Orbiter Claw");

    // Welcome state visible
    const welcome = page.locator("#welcome");
    await expect(welcome).toBeVisible();
    await expect(welcome).toContainText("Orbiter Claw");
    await expect(welcome).toContainText("local-first AI assistant");

    // Message input visible
    const input = page.locator("#message-input");
    await expect(input).toBeVisible();
    await expect(input).toHaveAttribute("placeholder", /Send a message/);

    // Send button visible but disabled
    const sendBtn = page.locator("#send-btn");
    await expect(sendBtn).toBeVisible();
    await expect(sendBtn).toBeDisabled();

    // Sidebar visible with title and new session button
    const sidebar = page.locator("#sidebar");
    await expect(sidebar).toContainText("Orbiter Claw");
    const newBtn = page.locator("#new-session-btn");
    await expect(newBtn).toBeVisible();

    // Theme toggle visible
    const themeToggle = page.locator("#theme-toggle");
    await expect(themeToggle).toBeVisible();

    // Connection status shown
    const status = page.locator("#connection-status");
    await expect(status).toBeVisible();
  });

  test("send button enables when text is entered", async ({ page }) => {
    await page.goto("/chat");

    const input = page.locator("#message-input");
    const sendBtn = page.locator("#send-btn");

    await expect(sendBtn).toBeDisabled();
    await input.fill("Hello");
    await expect(sendBtn).toBeEnabled();
    await input.fill("");
    await expect(sendBtn).toBeDisabled();
  });

  test("theme toggle switches dark mode", async ({ page }) => {
    await page.goto("/chat");

    const html = page.locator("html");
    const themeToggle = page.locator("#theme-toggle");

    // Default is light (no dark class unless OS prefers dark)
    await themeToggle.click();
    await expect(html).toHaveClass(/dark/);

    await themeToggle.click();
    await expect(html).not.toHaveClass(/dark/);
  });

  test("user message appears when Enter is pressed", async ({ page }) => {
    await page.goto("/chat");

    const input = page.locator("#message-input");
    await input.fill("Hello world");
    await input.press("Enter");

    // User message should appear in the messages area
    const messages = page.locator("#messages-inner");
    await expect(messages).toContainText("Hello world");

    // Welcome should be hidden
    const welcome = page.locator("#welcome");
    await expect(welcome).toBeHidden();
  });

  test("Shift+Enter adds newline instead of sending", async ({ page }) => {
    await page.goto("/chat");

    const input = page.locator("#message-input");
    await input.fill("Line 1");
    await input.press("Shift+Enter");
    await input.pressSequentially("Line 2");

    // Message should not have been sent — input still has content
    const value = await input.inputValue();
    expect(value).toContain("Line 1");
    expect(value).toContain("Line 2");
  });

  test("mobile sidebar toggle works", async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto("/chat");

    const sidebar = page.locator("#sidebar");
    const toggle = page.locator("#sidebar-toggle");

    // Sidebar should be off-screen on mobile
    await expect(sidebar).toHaveClass(/max-md:-translate-x-full/);

    // Click toggle to open
    await toggle.click();
    await expect(sidebar).not.toHaveClass(/max-md:-translate-x-full/);

    // Click backdrop to close
    const backdrop = page.locator("#sidebar-backdrop");
    await backdrop.click();
    await expect(sidebar).toHaveClass(/max-md:-translate-x-full/);
  });
});
