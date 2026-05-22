/**
 * Unified node export — importing this module registers all node types
 * into the NodeRegistry as a side effect.
 */

import { registerNodeType } from './registry'
import { DetectionNodeDefinition } from './DetectionNode/definition'
import { FilterNodeDefinition } from './FilterNode/definition'
import { ImageAnalysisNodeDefinition } from './ImageAnalysisNode/definition'
import { InputNodeDefinition } from './InputNode/definition'
import { MergeNodeDefinition } from './MergeNode/definition'
import { LogicNodeDefinition } from './LogicNode/definition'
import { RelationNodeDefinition } from './RelationNode/definition'
import { VLMNodeDefinition } from './VLMNode/definition'

// Register all node types
registerNodeType(DetectionNodeDefinition)
registerNodeType(FilterNodeDefinition)
registerNodeType(ImageAnalysisNodeDefinition)
registerNodeType(InputNodeDefinition)
registerNodeType(MergeNodeDefinition)
registerNodeType(LogicNodeDefinition)
registerNodeType(RelationNodeDefinition)
registerNodeType(VLMNodeDefinition)

// Re-export for convenience
export { DetectionNodeDefinition } from './DetectionNode/definition'
export { FilterNodeDefinition } from './FilterNode/definition'
export { ImageAnalysisNodeDefinition } from './ImageAnalysisNode/definition'
export { InputNodeDefinition } from './InputNode/definition'
export { MergeNodeDefinition } from './MergeNode/definition'
export { LogicNodeDefinition } from './LogicNode/definition'
export { RelationNodeDefinition } from './RelationNode/definition'
export { VLMNodeDefinition } from './VLMNode/definition'

export { registerNodeType, getNodeTypes, getNodeDefinition, buildReactFlowNodeTypes } from './registry'
export * from './types'
