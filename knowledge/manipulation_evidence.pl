:- use_module(library(semweb/rdf_db), [rdf_global_id/2, rdf_register_prefix/2]).
:- use_module(library(reasoner), [reasoner_define_relation/2]).

:- rdf_register_prefix(reassemble, 'https://example.org/reassemble/actions#').
:- reasoner_define_relation(manipulation_evidence, 5).

% manipulation_evidence(+Earlier, +Later, ?Action, ?Kind, ?Reason)
% The caller selects observations from inside the evaluated window.
manipulation_evidence(Earlier, Later, Action, Kind, Reason) :-
    ground(Earlier-Later),
    same_value(Earlier, Later, recordingId, _),
    same_value(Earlier, Later, clipId, _),
    same_value(Earlier, Later, observedGripper, _),
    snapshot_value(Earlier, observedAt, Start),
    snapshot_value(Later, observedAt, End),
    number(Start), number(End), Start < End,
    consistent_snapshot(Earlier),
    consistent_snapshot(Later),
    findall(evidence(Label, Type, Detail),
            observed_evidence(Earlier, Later, Label, Type, Detail), Observed),
    sort(Observed, Unique),
    select_evidence(Unique, Earlier, Later, evidence(ActionName, KindName, Reason)),
    rdf_global_id(reassemble:ActionName, Action),
    rdf_global_id(reassemble:KindName, Kind).

select_evidence([Evidence], _, _, Evidence).
select_evidence([], Earlier, Later, Evidence) :-
    findall(evidence(Action, attemptOnly, Reason),
            attempt_evidence(Earlier, Later, Action, Reason), Candidates),
    sort(Candidates, [Evidence]).

observed_evidence(Earlier, Later, pick, observedOutcome, liftedFromFreeSupport) :-
    same_value(Earlier, Later, observedPiece, _),
    same_value(Earlier, Later, observedSupport, _),
    snapshot_state(Earlier, supportStatus, onSupport),
    snapshot_state(Later, supportStatus, offSupport),
    snapshot_state(Earlier, gripStatus, unheld),
    snapshot_state(Later, gripStatus, held),
    \+ snapshot_state(Earlier, siteEngagement, seated),
    \+ snapshot_state(Later, siteEngagement, seated).
observed_evidence(Earlier, Later, place, observedOutcome, releasedOnFreeSupport) :-
    same_value(Earlier, Later, observedPiece, _),
    same_value(Earlier, Later, observedSupport, _),
    snapshot_state(Earlier, supportStatus, offSupport),
    snapshot_state(Later, supportStatus, onSupport),
    snapshot_state(Earlier, gripStatus, held),
    snapshot_state(Later, gripStatus, unheld),
    \+ snapshot_state(Earlier, siteEngagement, seated),
    \+ snapshot_state(Later, siteEngagement, seated).
observed_evidence(Earlier, Later, insert, observedOutcome, seatedInCompatibleSite) :-
    same_compatible_site(Earlier, Later),
    snapshot_state(Earlier, gripStatus, held),
    snapshot_state(Earlier, siteEngagement, separated),
    snapshot_state(Later, siteEngagement, seated).
observed_evidence(Earlier, Later, remove, observedOutcome, separatedFromCompatibleSiteWhileHeld) :-
    same_compatible_site(Earlier, Later),
    snapshot_state(Earlier, siteEngagement, seated),
    snapshot_state(Later, siteEngagement, separated),
    snapshot_state(Later, gripStatus, held).
observed_evidence(Earlier, Later, idle, observedInactivity, noTaskInteraction) :-
    interval_cue(Earlier, Later, noTaskInteraction),
    \+ known_state_change(Earlier, Later).

attempt_evidence(Earlier, Later, insert, contactWithoutObservedSeating) :-
    interval_cue(Earlier, Later, alignedContactTowardSite),
    same_compatible_site(Earlier, Later),
    snapshot_state(Earlier, gripStatus, held),
    snapshot_state(Earlier, siteEngagement, separated),
    \+ snapshot_state(Later, siteEngagement, seated).
attempt_evidence(Earlier, Later, remove, pullWithoutObservedSeparation) :-
    interval_cue(Earlier, Later, pullWhileGrippingAtSite),
    same_compatible_site(Earlier, Later),
    snapshot_state(Earlier, siteEngagement, seated),
    \+ snapshot_state(Later, siteEngagement, separated).

interval_cue(Earlier, Later, Cue) :-
    snapshot_value(Later, intervalStart, Earlier),
    snapshot_state(Later, intervalCue, Cue).

known_state_change(Earlier, Later) :-
    same_value(Earlier, Later, observedPiece, _),
    different_values(Earlier, Later, gripStatus).
known_state_change(Earlier, Later) :-
    same_value(Earlier, Later, observedPiece, _),
    same_value(Earlier, Later, observedSite, _),
    different_values(Earlier, Later, siteEngagement).
known_state_change(Earlier, Later) :-
    same_value(Earlier, Later, observedPiece, _),
    same_value(Earlier, Later, observedSupport, _),
    different_values(Earlier, Later, supportStatus).
known_state_change(Earlier, Later) :-
    different_values(Earlier, Later, gripperOccupancy).

same_compatible_site(Earlier, Later) :-
    same_value(Earlier, Later, observedPiece, Piece),
    same_value(Earlier, Later, observedSite, Site),
    manipulation_triple(Piece, compatibleSite, Site).

consistent_snapshot(Snapshot) :-
    \+ (member(Property, [observedPiece, observedSite, observedSupport,
                          gripStatus, gripperOccupancy, siteEngagement, supportStatus,
                          intervalStart, intervalCue]),
        setof(Value, manipulation_triple(Snapshot, Property, Value), [_, _|_])),
    \+ (snapshot_state(Snapshot, gripStatus, held),
        snapshot_state(Snapshot, gripperOccupancy, empty)),
    \+ (snapshot_state(Snapshot, siteEngagement, seated),
        snapshot_state(Snapshot, supportStatus, onSupport)).

same_value(Earlier, Later, Property, Value) :-
    snapshot_value(Earlier, Property, Value),
    snapshot_value(Later, Property, Value).

different_values(Earlier, Later, Property) :-
    snapshot_value(Earlier, Property, Before),
    snapshot_value(Later, Property, After),
    Before \== After.

snapshot_state(Snapshot, Property, State) :-
    rdf_global_id(reassemble:State, Iri),
    snapshot_value(Snapshot, Property, Iri).

snapshot_value(Snapshot, Property, Value) :-
    setof(Observed, manipulation_triple(Snapshot, Property, Observed), [Value]).

manipulation_triple(Subject, Property, Value) :-
    rdf_global_id(reassemble:Property, Predicate),
    % Decode RDF literals with a fresh variable before comparing their values.
    triple(Subject, Predicate, Observed),
    Value = Observed.
