from nexttransformer.methods.chordres_ambient_lookup import AmbientChord, frozen as frozen


class ZeroStartChord(AmbientChord):
    initial_velocity_mode = "zero"


model = ZeroStartChord
