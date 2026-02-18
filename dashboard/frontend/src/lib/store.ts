import { create } from 'zustand'

type ScenarioUiState = {
  t: number
  tMin: number
  tMax: number
  isPlaying: boolean
  productFilter: string
  minFlow: number
  topK: number
  setBounds: (tMin: number, tMax: number) => void
  setT: (t: number) => void
  stepForward: () => void
  stepBackward: () => void
  setPlaying: (playing: boolean) => void
  setProductFilter: (value: string) => void
  setMinFlow: (value: number) => void
  setTopK: (value: number) => void
}

export const useScenarioUiStore = create<ScenarioUiState>((set, get) => ({
  t: 0,
  tMin: 0,
  tMax: 0,
  isPlaying: false,
  productFilter: '',
  minFlow: 0,
  topK: 10,
  setBounds: (tMin, tMax) =>
    set((state) => ({
      tMin,
      tMax,
      t: Math.min(Math.max(state.t, tMin), tMax),
    })),
  setT: (t) => {
    const { tMin, tMax } = get()
    set({ t: Math.min(Math.max(t, tMin), tMax) })
  },
  stepForward: () => {
    const { t, tMax } = get()
    set({ t: Math.min(t + 1, tMax) })
  },
  stepBackward: () => {
    const { t, tMin } = get()
    set({ t: Math.max(t - 1, tMin) })
  },
  setPlaying: (playing) => set({ isPlaying: playing }),
  setProductFilter: (value) => set({ productFilter: value }),
  setMinFlow: (value) => set({ minFlow: value }),
  setTopK: (value) => set({ topK: Math.max(1, Math.min(50, value)) }),
}))
