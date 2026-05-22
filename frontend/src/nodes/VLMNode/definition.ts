import type { NodeDefinition } from '../types'
import { PortType } from '../types'
import { VLMNodeComponent } from './index'

export const VLMNodeDefinition: NodeDefinition = {
  type: 'vlm',
  label: 'VLM',
  description: 'Send images to a Vision Language Model API and parse detections from the response',
  inputPorts: [
    {
      name: 'input',
      portType: PortType.ImageStream,
      label: 'Images',
      description: 'Input image frames (List[np.ndarray])',
    },
  ],
  outputPorts: [
    {
      name: 'output',
      portType: PortType.ObjectStream,
      label: 'Objects',
      description: 'Objects parsed from the VLM structured response',
    },
  ],
  defaultConfig: {
    model: 'gpt-4o-mini',
    prompt:
      'List every visible object. Return JSON: {"objects": [{"class_name": str, "x": int, "y": int, "w": int, "h": int, "confidence": float}]}',
    apiKeyEnv: 'OPENAI_API_KEY',
    maxTokens: 1024,
  },
  component: VLMNodeComponent,
}
