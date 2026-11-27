from nexttransformer.methods.chordres_ambient_lookup import AmbientChord, frozen as frozen


class ProjectedH0Chord(AmbientChord):
    initial_velocity_mode = "projected_h0"


model = ProjectedH0Chord
