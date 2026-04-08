use egui::{Pos2, Rect};
use egui_snarl::{InPinId, NodeId, OutPinId};

use crate::data::loaded_files::VolumeColormap;
use crate::data::trx_data::RenderStyle;
use crate::renderer::mesh_renderer::SurfaceColormap;

use super::*;

pub fn make_node(document: &mut WorkflowDocument, kind: WorkflowNodeKind, pos: Pos2) -> NodeId {
    let uuid = WorkflowNodeUuid(document.next_node_uuid);
    document.next_node_uuid += 1;
    document.graph.insert_node(
        pos,
        WorkflowNode {
            uuid,
            label: kind.title().to_string(),
            kind,
        },
    )
}

pub fn suggest_asset_branch_origin(document: &WorkflowDocument) -> Pos2 {
    let mut min_x = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    for (pos, _) in document.graph.nodes_pos() {
        min_x = min_x.min(pos.x);
        max_y = max_y.max(pos.y);
    }

    if min_x.is_finite() && max_y.is_finite() {
        Pos2::new(min_x, max_y + 170.0)
    } else {
        Pos2::new(40.0, 80.0)
    }
}

fn branch_bounds(document: &WorkflowDocument, nodes: &[NodeId]) -> Rect {
    let mut bounds = Rect::NOTHING;
    for node_id in nodes {
        if let Some(node) = document.graph.get_node_info(*node_id) {
            bounds.extend_with(node.pos);
        }
    }
    if bounds.is_finite() {
        bounds.expand2(egui::vec2(220.0, 120.0))
    } else {
        Rect::from_min_size(Pos2::ZERO, egui::vec2(640.0, 240.0))
    }
}

pub fn add_default_nodes_for_asset(
    document: &mut WorkflowDocument,
    asset: &WorkflowAssetDocument,
    pos: Pos2,
    streamline_limit: Option<usize>,
) -> SeededWorkflowBranch {
    match asset {
        WorkflowAssetDocument::Streamlines { id, .. } => {
            let source = make_node(
                document,
                WorkflowNodeKind::StreamlineSource { source_id: *id },
                pos,
            );
            let limit = make_node(
                document,
                WorkflowNodeKind::LimitStreamlines {
                    limit: streamline_limit.unwrap_or(30_000).max(1),
                    randomize: false,
                    seed: 1,
                },
                pos + egui::vec2(240.0, 0.0),
            );
            let display = make_node(
                document,
                WorkflowNodeKind::StreamlineDisplay {
                    enabled: true,
                    render_style: RenderStyle::Flat,
                    tube_radius_mm: 0.4,
                    tube_sides: 8,
                    slab_half_width_mm: 5.0,
                },
                pos + egui::vec2(480.0, 0.0),
            );
            document.graph.connect(
                OutPinId {
                    node: source,
                    output: 0,
                },
                InPinId {
                    node: limit,
                    input: 0,
                },
            );
            document.graph.connect(
                OutPinId {
                    node: limit,
                    output: 0,
                },
                InPinId {
                    node: display,
                    input: 0,
                },
            );
            SeededWorkflowBranch {
                bounds: branch_bounds(document, &[source, limit, display]),
                primary_selection: WorkflowSelection::Node(document.graph[limit].uuid),
            }
        }
        WorkflowAssetDocument::Volume { id, .. } => {
            let source = make_node(
                document,
                WorkflowNodeKind::VolumeSource { source_id: *id },
                pos,
            );
            let display = make_node(
                document,
                WorkflowNodeKind::VolumeDisplay {
                    colormap: VolumeColormap::Grayscale,
                    opacity: 1.0,
                    window_center: 0.5,
                    window_width: 1.0,
                },
                pos + egui::vec2(220.0, 0.0),
            );
            document.graph.connect(
                OutPinId {
                    node: source,
                    output: 0,
                },
                InPinId {
                    node: display,
                    input: 0,
                },
            );
            SeededWorkflowBranch {
                bounds: branch_bounds(document, &[source, display]),
                primary_selection: WorkflowSelection::Node(document.graph[source].uuid),
            }
        }
        WorkflowAssetDocument::Surface { id, .. } => {
            let source = make_node(
                document,
                WorkflowNodeKind::SurfaceSource { source_id: *id },
                pos,
            );
            let display = make_node(
                document,
                WorkflowNodeKind::SurfaceDisplay {
                    color: DEFAULT_SURFACE_COLOR,
                    opacity: DEFAULT_SURFACE_OPACITY,
                    outline_color: DEFAULT_SURFACE_COLOR,
                    outline_thickness: 1.25,
                    show_projection_map: false,
                    map_opacity: 1.0,
                    map_threshold: 0.0,
                    gloss: 0.45,
                    projection_colormap: SurfaceColormap::Inferno,
                    range_min: 0.0,
                    range_max: 1.0,
                },
                pos + egui::vec2(220.0, 0.0),
            );
            document.graph.connect(
                OutPinId {
                    node: source,
                    output: 0,
                },
                InPinId {
                    node: display,
                    input: 0,
                },
            );
            SeededWorkflowBranch {
                bounds: branch_bounds(document, &[source, display]),
                primary_selection: WorkflowSelection::Node(document.graph[source].uuid),
            }
        }
        WorkflowAssetDocument::Parcellation { id, .. } => {
            let source = make_node(
                document,
                WorkflowNodeKind::ParcellationSource { source_id: *id },
                pos,
            );
            let display = make_node(
                document,
                WorkflowNodeKind::ParcellationDisplay {
                    labels_csv: String::new(),
                    opacity: 0.9,
                },
                pos + egui::vec2(240.0, 0.0),
            );
            document.graph.connect(
                OutPinId {
                    node: source,
                    output: 0,
                },
                InPinId {
                    node: display,
                    input: 0,
                },
            );
            SeededWorkflowBranch {
                bounds: branch_bounds(document, &[source, display]),
                primary_selection: WorkflowSelection::Node(document.graph[source].uuid),
            }
        }
    }
}
