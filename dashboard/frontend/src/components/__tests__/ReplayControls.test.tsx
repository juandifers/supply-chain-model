import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, beforeEach } from 'vitest'
import { ReplayControls } from '../ReplayControls'
import { useScenarioUiStore } from '../../lib/store'

const kpis = [
  { t: 0, shock_exposure: 0.1 },
  { t: 1, shock_exposure: 0.3 },
  { t: 2, shock_exposure: 0.2 },
] as unknown as Array<any>

describe('ReplayControls', () => {
  beforeEach(() => {
    useScenarioUiStore.setState({
      t: 0,
      tMin: 0,
      tMax: 2,
      isPlaying: false,
      productFilter: '',
      minFlow: 0,
      topK: 10,
    })
  })

  it('jumps to shock peak', async () => {
    const user = userEvent.setup()
    render(<ReplayControls kpis={kpis as any} />)

    await user.click(screen.getByRole('button', { name: /jump to shock peak/i }))

    expect(useScenarioUiStore.getState().t).toBe(1)
  })
})
