import ReactECharts from 'echarts-for-react'
import type { KPIRow } from '../types'

type KpiTimelineChartProps = {
  kpis: KPIRow[]
  activeT: number
}

export function KpiTimelineChart({ kpis, activeT }: KpiTimelineChartProps) {
  const t = kpis.map((x) => x.t)

  const option = {
    animationDuration: 500,
    tooltip: { trigger: 'axis' },
    legend: { top: 8, textStyle: { color: '#ece6d7' } },
    grid: [{ left: 56, right: 24, top: 40, height: '27%' }, { left: 56, right: 24, top: '42%', height: '26%' }, { left: 56, right: 24, top: '72%', height: '22%' }],
    xAxis: [
      { type: 'category', gridIndex: 0, data: t, axisLabel: { color: '#d8d3c5' }, axisLine: { lineStyle: { color: '#5b615f' } } },
      { type: 'category', gridIndex: 1, data: t, axisLabel: { color: '#d8d3c5' }, axisLine: { lineStyle: { color: '#5b615f' } } },
      { type: 'category', gridIndex: 2, data: t, axisLabel: { color: '#d8d3c5' }, axisLine: { lineStyle: { color: '#5b615f' } } },
    ],
    yAxis: [
      { type: 'value', gridIndex: 0, axisLabel: { color: '#d8d3c5' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } } },
      { type: 'value', gridIndex: 1, axisLabel: { color: '#d8d3c5' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } } },
      { type: 'value', gridIndex: 2, min: 0, max: 1, axisLabel: { color: '#d8d3c5' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } } },
    ],
    series: [
      {
        name: 'Transactions',
        type: 'line',
        smooth: true,
        xAxisIndex: 0,
        yAxisIndex: 0,
        data: kpis.map((x) => x.transactions),
        lineStyle: { color: '#42e2b8', width: 2 },
      },
      {
        name: 'Open Orders',
        type: 'line',
        smooth: true,
        xAxisIndex: 0,
        yAxisIndex: 0,
        data: kpis.map((x) => x.open_orders),
        lineStyle: { color: '#f69468', width: 2 },
      },
      {
        name: 'Backlog Units',
        type: 'line',
        smooth: true,
        xAxisIndex: 1,
        yAxisIndex: 1,
        data: kpis.map((x) => x.consumer_backlog_units),
        lineStyle: { color: '#ff5f6d', width: 2 },
      },
      {
        name: 'Shock Exposure',
        type: 'line',
        smooth: true,
        xAxisIndex: 2,
        yAxisIndex: 2,
        data: kpis.map((x) => x.shock_exposure),
        lineStyle: { color: '#ffd166', width: 2 },
      },
      {
        name: 'Cum Fill Rate',
        type: 'line',
        smooth: true,
        xAxisIndex: 2,
        yAxisIndex: 2,
        data: kpis.map((x) => x.consumer_cumulative_fill_rate),
        lineStyle: { color: '#7b8cff', width: 2 },
      },
      {
        name: 'Active Timestep',
        type: 'line',
        xAxisIndex: 2,
        yAxisIndex: 2,
        data: t.map((x) => (x === activeT ? 1 : 0)),
        lineStyle: { width: 0 },
        symbol: 'diamond',
        symbolSize: 10,
        itemStyle: { color: '#ffffff' },
      },
    ],
  }

  return (
    <section className="panel chart-panel">
      <h3>Timeline KPIs</h3>
      <ReactECharts option={option} style={{ width: '100%', height: 360 }} />
    </section>
  )
}
