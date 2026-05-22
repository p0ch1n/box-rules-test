import { Handle, Position } from 'reactflow'
import type { NodeComponentProps } from '../types'
import { PORT_TYPE_COLORS, PortType } from '../types'
import { usePipelineStore } from '@/store/pipelineStore'
import { NodeHelp } from '../NodeHelp'

const MODELS = ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo'] as const
type VLMModel = typeof MODELS[number]

const HELP_LINES = [
  'Sends each input image to a Vision Language Model API.',
  'The model returns structured JSON; detected objects are parsed into ObjectStream.',
  'Set the API key in the environment variable named in "API key env".',
  'Left port (purple): ImageStream input.',
  'Right port (blue): ObjectStream output.',
  'Prompt must instruct the model to return {"objects": [...]} JSON.',
]

interface VLMConfig {
  model: VLMModel
  prompt: string
  apiKeyEnv: string
  maxTokens: number
}

export function VLMNodeComponent({ id, data, selected }: NodeComponentProps) {
  const updateNodeConfig = usePipelineStore((s) => s.updateNodeConfig)
  const config = data.config as unknown as VLMConfig

  const setConfig = (patch: Partial<VLMConfig>) =>
    updateNodeConfig(id, { ...config, ...patch } as Record<string, unknown>)

  const isInvalid = !config.apiKeyEnv?.trim() || !config.model

  return (
    <div
      style={{
        background: '#fdf4ff',
        border: selected
          ? '2px solid #9333ea'
          : isInvalid
          ? '2px solid #f87171'
          : '2px solid #d8b4fe',
        borderRadius: 8,
        padding: '12px 16px',
        minWidth: 280,
        fontSize: 13,
        boxShadow: '0 2px 8px rgba(0,0,0,0.12)',
      }}
    >
      <Handle
        type="target"
        position={Position.Left}
        id="input"
        style={{ background: PORT_TYPE_COLORS[PortType.ImageStream], width: 12, height: 12 }}
      />

      <div
        style={{
          fontWeight: 700,
          marginBottom: 8,
          color: '#7e22ce',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        VLM
        {isInvalid && (
          <span style={{ color: '#dc2626', fontSize: 11, fontWeight: 400 }}>
            ⚠ config required
          </span>
        )}
      </div>

      <label style={{ display: 'block', marginBottom: 6 }}>
        Model
        <select
          value={config.model}
          onChange={(e) => setConfig({ model: e.target.value as VLMModel })}
          style={{ display: 'block', width: '100%', marginTop: 2 }}
        >
          {MODELS.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
      </label>

      <label style={{ display: 'block', marginBottom: 6 }}>
        Prompt
        <textarea
          value={config.prompt}
          onChange={(e) => setConfig({ prompt: e.target.value })}
          rows={3}
          style={{ display: 'block', width: '100%', marginTop: 2, resize: 'vertical', fontSize: 12 }}
        />
      </label>

      <label style={{ display: 'block', marginBottom: 6 }}>
        API key env
        <input
          value={config.apiKeyEnv}
          onChange={(e) => setConfig({ apiKeyEnv: e.target.value })}
          placeholder="OPENAI_API_KEY"
          style={{
            display: 'block',
            width: '100%',
            marginTop: 2,
            borderColor: !config.apiKeyEnv?.trim() ? '#f87171' : undefined,
          }}
        />
      </label>

      <label style={{ display: 'block', marginBottom: 4 }}>
        Max tokens
        <input
          type="number"
          min={1}
          max={4096}
          value={config.maxTokens}
          onChange={(e) => {
            const v = parseInt(e.target.value, 10)
            if (!isNaN(v) && v >= 1) setConfig({ maxTokens: v })
          }}
          style={{ display: 'block', width: '100%', marginTop: 2 }}
        />
      </label>

      <NodeHelp lines={HELP_LINES} accentColor="#7e22ce" />

      <Handle
        type="source"
        position={Position.Right}
        id="output"
        style={{ background: PORT_TYPE_COLORS[PortType.ObjectStream], width: 12, height: 12 }}
      />
    </div>
  )
}
