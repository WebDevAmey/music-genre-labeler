"""The labeling task: the only task-specific file. Change it to label something else."""

NAME = "Music genre from caption"

LABELS: tuple[str, ...] = (
    "rock",
    "pop",
    "electronic",
    "hip_hop",
    "classical",
    "jazz_blues",
    "folk_country",
    "world",
    "other",
)

GUIDELINES = """\
Pick the single genre that best fits the music described in the caption.

- rock: rock, metal, punk, grunge, indie rock; distorted guitars and drum kit.
- pop: mainstream pop, ballads, R&B, soul, k-pop, disco; vocal-led and catchy.
- electronic: EDM, house, techno, dubstep, trance, synthwave, ambient electronic.
- hip_hop: rap, trap, boom bap, lo-fi hip hop beats.
- classical: orchestral, chamber, solo classical piano or strings, opera, film score.
- jazz_blues: jazz, swing, bossa nova, blues, funk.
- folk_country: folk, country, bluegrass, acoustic singer-songwriter.
- world: traditional or regional music (e.g. Indian classical, Arabic, Latin folk,
  African, Celtic, Chinese traditional), reggae, flamenco.
- other: none of the above, or not enough information (e.g. a lone sound effect,
  a children's song, a religious chant, a jingle).

When two genres fit, pick the one the caption emphasises most.\
"""
