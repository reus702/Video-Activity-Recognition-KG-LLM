"""Live integration check. Run with the KnowRob Python binding available."""

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PREFIXES = {
    "reassemble": "https://example.org/reassemble/actions",
    "soma": "http://www.ease-crc.org/ont/SOMA.owl",
    "observation": "https://example.org/reassemble/observations/bnc",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ontology_dir", type=Path, help="Directory with DUL.owl and SOMA.owl")
    args = parser.parse_args()

    flags = sys.getdlopenflags()
    try:
        sys.setdlopenflags(flags | os.RTLD_GLOBAL)
        import knowrob
    finally:
        sys.setdlopenflags(flags)

    sources = [
        (args.ontology_dir / "DUL.owl", "rdf-xml"),
        (args.ontology_dir / "SOMA.owl", "rdf-xml"),
        (ROOT / "knowledge/reassemble_actions.ttl", "turtle"),
        (ROOT / "knowledge/reassemble_manipulation_state.ttl", "turtle"),
        (ROOT / "knowledge/reassemble_catalog.ttl", "turtle"),
        (ROOT / "knowledge/reassemble_bnc_state.ttl", "turtle"),
        (ROOT / "tests/fixtures/bnc_snapshots.ttl", "turtle"),
        (ROOT / "tests/fixtures/manipulation_snapshots.ttl", "turtle"),
        (ROOT / "tests/fixtures/manipulation_evidence.ttl", "turtle"),
        (ROOT / "knowledge/observations/bnc_validation_window.ttl", "turtle"),
    ]
    for path, _ in sources:
        if not path.is_file():
            parser.error(f"Missing ontology: {path}")

    settings = {
        "semantic-web": {"prefixes": [
            {"alias": alias, "uri": uri} for alias, uri in PREFIXES.items()
        ]},
        "data-sources": [{"path": str(path.resolve()), "format": fmt} for path, fmt in sources],
        "data-backends": [{"name": "pl", "type": "Prolog:rdf_db"}],
        "reasoner": [{
            "name": "pl", "type": "Prolog", "data-backend": "pl",
            "imports": [{"path": str(ROOT / "knowledge/manipulation_evidence.pl"), "format": "prolog"}],
        }],
    }
    kb = knowrob.KnowledgeBase(json.dumps(settings))

    def query(expression, expected):
        formula = knowrob.QueryParser.parse(expression)
        context = knowrob.QueryContext(knowrob.QueryFlag.QUERY_FLAG_ALL_SOLUTIONS)
        stream = kb.submitQuery(formula, context)
        queue = stream.createQueue()
        answer = queue.pop_front()
        if expected is None:
            if isinstance(answer, knowrob.AnswerYes):
                raise AssertionError(f"{expression}: unexpected evidence {answer}")
            print(f"{expression} -> no action evidence", flush=True)
            return
        if not isinstance(answer, knowrob.AnswerYes):
            raise AssertionError(f"{expression}: {answer}")
        bindings = answer.substitution()
        actual = {name: str(bindings.get(name)) for name in expected}
        if actual != expected:
            raise AssertionError(f"{expression}: expected {expected}, got {actual}")
        print(f"{expression} -> {actual}", flush=True)

    query(
        "reassemble:observedConnector(reassemble:afterRelease, Connector), "
        "reassemble:observedSocket(reassemble:afterRelease, Socket), "
        "reassemble:compatibleSocket(Connector, Socket), "
        "soma:hasPhysicalComponent(Station, Socket), "
        "reassemble:observedGripper(reassemble:afterRelease, Gripper), "
        "soma:hasPhysicalComponent(Arm, Gripper)",
        {"Connector": "reassemble:bnc", "Socket": "reassemble:bncSocket",
         "Station": "reassemble:bncWorkstation", "Arm": "reassemble:robotArm",
         "Gripper": "reassemble:gripper"},
    )
    query(
        "reassemble:gripStatus(reassemble:beforeRelease, Before), "
        "reassemble:gripStatus(reassemble:afterRelease, After), "
        "reassemble:socketEngagement(reassemble:beforeRelease, Engagement), "
        "reassemble:socketEngagement(reassemble:afterRelease, Engagement), "
        "reassemble:gripperOpening(reassemble:afterRelease, Opening)",
        {"Before": "reassemble:held", "After": "reassemble:unheld",
         "Engagement": "reassemble:seated", "Opening": "reassemble:open"},
    )
    query(
        "reassemble:socketEngagement(observation:'2025-01-09-14-57-02_hand_clip_0021_last', Status)",
        {"Status": "reassemble:seated"},
    )
    catalog_pairs = [
        ("bnc", "bncSocket"), ("dSub", "dSubSocket"), ("ethernet", "ethernetSocket"),
        ("usb", "usbSocket"), ("waterproof", "waterproofSocket"), ("m16Nut", "m16NutMount"),
        *((f"{size}Gear", f"{size}GearMount") for size in ("small", "medium", "large")),
        *((f"{shape}Peg{size}", f"{shape}PegHole{size}")
          for shape in ("round", "square") for size in range(1, 5)),
    ]
    for piece, site in catalog_pairs:
        query(f"reassemble:compatibleSite(reassemble:{piece}, Site)",
              {"Site": f"reassemble:{site}"})

    def check_snapshot(snapshot, piece, site):
        query(
            f"reassemble:observedPiece({snapshot}, Piece), "
            f"reassemble:observedSite({snapshot}, Site), "
            f"reassemble:siteEngagement({snapshot}, Status), "
            "reassemble:compatibleSite(Piece, Site)",
            {"Piece": f"reassemble:{piece}", "Site": f"reassemble:{site}",
             "Status": "reassemble:seated"},
        )

    check_snapshot("observation:'2025-01-09-14-57-02_hand_clip_0021_last'", "bnc", "bncSocket")
    check_snapshot("reassemble:pegAtSite", "roundPeg4", "roundPegHole4")
    query(
        "reassemble:observedSupport(reassemble:gearOnTable, Surface), "
        "reassemble:supportStatus(reassemble:gearOnTable, Status)",
        {"Surface": "reassemble:reviewTable", "Status": "reassemble:onSupport"},
    )
    for earlier, later, action, kind, reason in [
        ("gearOnTable", "gearLifted", "pick", "observedOutcome", "liftedFromFreeSupport"),
        ("gearLifted", "gearBackOnTable", "place", "observedOutcome", "releasedOnFreeSupport"),
        ("usbInGripper", "usbInserted", "insert", "observedOutcome", "seatedInCompatibleSite"),
        ("pegAtSite", "pegRemoved", "remove", "observedOutcome", "separatedFromCompatibleSiteWhileHeld"),
        ("beforeRelease", "bncTakenOut", "remove", "observedOutcome", "separatedFromCompatibleSiteWhileHeld"),
        ("emptyAtStart", "emptyWithIdleCue", "idle", "observedInactivity", "noTaskInteraction"),
        ("usbInGripper", "usbContact", "insert", "attemptOnly", "contactWithoutObservedSeating"),
        ("pegGrasped", "pegPulled", "remove", "attemptOnly", "pullWithoutObservedSeparation"),
    ]:
        query(
            f"manipulation_evidence(reassemble:{earlier}, reassemble:{later}, Action, Kind, Reason)",
            {"Action": f"reassemble:{action}", "Kind": f"reassemble:{kind}", "Reason": reason},
        )
    for earlier, later in [
        ("beforeRelease", "afterRelease"),
        ("pegAtSite", "pegGrasped"),
        ("pegAtSite", "pegPulled"),
        ("pegAtSite", "pegEngagementConflicts"),
        ("usbInserted", "usbInGripper"),
        ("usbInGripper", "usbOtherClip"),
        ("nutAtWrongSite", "nutStillAtWrongSite"),
        ("emptyAtStart", "emptyWithoutCue"),
    ]:
        query(
            f"manipulation_evidence(reassemble:{earlier}, reassemble:{later}, Action, Kind, Reason)",
            None,
        )
    query(
        "manipulation_evidence(observation:'2025-01-09-14-57-02_hand_clip_0021_first', "
        "observation:'2025-01-09-14-57-02_hand_clip_0021_last', Action, Kind, Reason)",
        None,
    )


if __name__ == "__main__":
    main()
