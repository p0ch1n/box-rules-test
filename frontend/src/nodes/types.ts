/**
 * Core type system for the node registry and port validation layer.
 */

// ------------------------------------------------------------------ //
// Port Type Enum — wire compatibility system
// ------------------------------------------------------------------ //

export enum PortType {
  /** Single-branch stream of BBox objects */
  BoxStream = 'BoxStream',
  /** Multi-branch aggregate of BBoxes (preserves lineage) */
  Collection = 'Collection',
  /** Boolean trigger with attached metadata dict */
  LogicSignal = 'LogicSignal',
  /** Raw image frames fed into detection nodes */
  ImageStream = 'ImageStream',
  /** Image frames paired with their BBoxes — output of DetectionNode / ImageAnalysisNode */
  AnnotatedStream = 'AnnotatedStream',
  /**
   * Static reference images that do not change frame-to-frame.
   * Used for template matching references, background models, etc.
   * A node that outputs this type declares no input ports (self-seeding).
   */
  ReferenceImageStream = 'ReferenceImageStream',
}

/**
 * Canonical handle colors for each PortType.
 * Always reference this map in node components — do not hardcode color strings.
 *
 * @example
 * <Handle style={{ background: PORT_TYPE_COLORS[PortType.BoxStream] }} />
 */
export const PORT_TYPE_COLORS: Record<PortType, string> = {
  [PortType.BoxStream]: '#3b82f6',           // blue
  [PortType.Collection]: '#f59e0b',          // amber
  [PortType.LogicSignal]: '#22c55e',         // green
  [PortType.ImageStream]: '#7c3aed',         // purple
  [PortType.AnnotatedStream]: '#f97316',     // orange
  [PortType.ReferenceImageStream]: '#e11d48', // rose
}

/**
 * Connection rules.
 * BoxStream → Collection is allowed so that a single-stream node (FilterNode,
 * RelationNode, …) can feed directly into a LogicNode without a Merge step.
 * All image-carrying types are self-contained and cannot cross-connect.
 */
export const PORT_TYPE_COMPATIBILITY: Record<PortType, PortType[]> = {
  [PortType.BoxStream]: [PortType.BoxStream, PortType.Collection],
  [PortType.Collection]: [PortType.Collection],
  [PortType.LogicSignal]: [PortType.LogicSignal],
  [PortType.ImageStream]: [PortType.ImageStream],
  [PortType.AnnotatedStream]: [PortType.AnnotatedStream],
  [PortType.ReferenceImageStream]: [PortType.ReferenceImageStream],
}

// ------------------------------------------------------------------ //
// Port Definition
// ------------------------------------------------------------------ //

export interface PortDefinition {
  name: string
  portType: PortType
  label?: string
  description?: string
  optional?: boolean
}

// ------------------------------------------------------------------ //
// Node Definition (registry entry)
// ------------------------------------------------------------------ //

export interface NodeDefinition {
  /** Unique type string matching the JSON schema "type" field */
  type: string
  /** Display label shown in the canvas */
  label: string
  /** Short description shown in tooltip / sidebar */
  description: string
  /** Input port declarations */
  inputPorts: PortDefinition[]
  /** Output port declarations */
  outputPorts: PortDefinition[]
  /** Default config values for new nodes */
  defaultConfig: Record<string, unknown>
  /** React component to render this node (lazy-resolved at runtime) */
  component: React.ComponentType<NodeComponentProps>
}

// ------------------------------------------------------------------ //
// Node Component Props (all node components share this interface)
// ------------------------------------------------------------------ //

export interface NodeComponentProps {
  id: string
  data: NodeData
  selected: boolean
}

export interface NodeData {
  label: string
  nodeType: string
  config: Record<string, unknown>
  /** Available class labels inferred from upstream nodes */
  availableClasses?: string[]
}

// ------------------------------------------------------------------ //
// Pipeline JSON types (mirror of Pydantic models)
// ------------------------------------------------------------------ //

export interface PipelinePosition {
  x: number
  y: number
}

export interface PipelineNode {
  id: string
  type: string
  position: PipelinePosition
  config: Record<string, unknown>
}

export interface PipelineEdge {
  id: string
  source: string
  source_port: string
  target: string
  target_port: string
}

export interface PipelineMetadata {
  tool_id: string
  class_list: string[]
  description?: string
}

export interface PipelineJSON {
  version: string
  metadata: PipelineMetadata
  nodes: PipelineNode[]
  edges: PipelineEdge[]
}
