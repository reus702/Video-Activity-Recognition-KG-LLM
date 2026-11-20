import argparse
import hashlib
import json
from pathlib import Path


ACTIONS = ("PICK", "INSERT", "REMOVE", "PLACE")
NAMESPACE = "https://example.org/reassemble/actions#"
ONTOLOGY = Path(__file__).resolve().parents[1] / "knowledge/reassemble_actions.ttl"


def query_transitions():
    '''Collect positive KnowRob answers for the four actions.'''
    try:
        import knowrob
    except ImportError as exc:
        raise RuntimeError("Run the exporter with a Python interpreter that can import KnowRob 2") from exc

    settings = {
        "semantic-web": {"prefixes": [{"alias": "reassemble", "uri": NAMESPACE.rstrip("#")}]},
        "data-sources": [{"path": str(ONTOLOGY), "format": "turtle"}],
        "data-backends": [{"name": "pl", "type": "Prolog:rdf_db"}],
        "reasoner": [{"name": "pl", "type": "Prolog", "data-backend": "pl"}],
    }
    knowledge_base = knowrob.KnowledgeBase(json.dumps(settings))
    transitions = []
    for previous in ACTIONS:
        for candidate in ACTIONS:
            query = (f"reassemble:preferredSuccessor(reassemble:{previous.lower()}, "
                     f"reassemble:{candidate.lower()})")
            formula = knowrob.QueryParser.parse(query)
            context = knowrob.QueryContext(knowrob.QueryFlag.QUERY_FLAG_ALL_SOLUTIONS)
            stream = knowledge_base.submitQuery(formula, context)
            answer = stream.createQueue().pop_front()
            if isinstance(answer, knowrob.AnswerYes):
                transitions.append({"previous": previous, "candidate": candidate, "query": query})
    if not transitions:
        raise RuntimeError("KnowRob returned no positive transitions. Check that it loaded the ontology.")
    return {"schema_version": 1, "source": "KnowRob",
            "ontology_sha256": hashlib.sha256(ONTOLOGY.read_bytes()).hexdigest(),
            "transitions": transitions}


def main():
    parser = argparse.ArgumentParser(description="Export transition evidence from KnowRob 2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    try:
        evidence = query_transitions()
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, indent=2)
    print(f"Saved {len(evidence['transitions'])} KnowRob answers to {args.output}")


if __name__ == "__main__":
    main()
