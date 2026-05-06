from landscape.retrieval.query import EntityPath


def render_path(path: EntityPath) -> str:
    """
    Render an EntityPath as a human-readable string.

    Examples:
        - Direct hit (no edges): "(Qdrant) [direct match]"
        - One hop: "(Eric) -[DISCUSSION]-> Netflix [TECHNOLOGY]"
        - Two hops: "(Project Aurora) -[USES]-> PostgreSQL [TECHNOLOGY] -[APPROVED_BY]-> Maya Chen [PERSON]"
        - Negated edge: "(Alice) -[NOT WORKS_FOR]-> Acme Corp [ORGANIZATION]"
        - Empty path: "(unknown)"
    """
    if not path.nodes:
        return "(unknown)"
    if not path.edges:
        return f"({path.nodes[0].name}) [direct match]"

    parts = [f"({path.nodes[0].name})"]
    for edge, node in zip(path.edges, path.nodes[1:]):
        label = f"NOT {edge.type}" if edge.negated else edge.type
        parts.append(f"-[{label}]->")
        parts.append(f"{node.name} [{node.type}]")

    return " ".join(parts)
