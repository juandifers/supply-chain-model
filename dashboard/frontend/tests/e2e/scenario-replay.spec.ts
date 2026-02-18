import { test, expect } from '@playwright/test'

test('opens home and shows scenario section', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: /supplysim control tower/i })).toBeVisible()
  await expect(page.getByRole('heading', { name: /scenarios/i })).toBeVisible()
})
