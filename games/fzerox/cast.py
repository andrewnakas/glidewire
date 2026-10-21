"""Glidewire's cast: our own pilot designs (2026-10 originality rebuild). Keys are the decomp's texture slot
names (which picture slot a pilot fills); everything else is ours. Names live in names.json.
Template options: see pilots.py (human face options, or monkey / lizard / rex / robot / ead / octo / alien /
skull / draq / twin / dog / zero heads)."""

CHARACTERS = {
    # Rook Valdane: short black hair with shaved sides, green wrap visor, teal and white suit, calm
    "CaptainFalcon": dict(t="human", skin=(206, 150, 112), style="short", hair=(28, 28, 34), eyes="visor",
                          iris=(60, 230, 120), mouth="closed", jaw=1.05, suit=(20, 130, 130), collar=(236, 236, 240),
                          bg=(16, 46, 50)),
    "CaptainFalconAlt": dict(t="human", skin=(206, 150, 112), style="short", hair=(28, 28, 34), eyes="normal",
                             iris=(70, 120, 60), mouth="smile", jaw=1.05, suit=(236, 236, 240), collar=(20, 130, 130),
                             bg=(20, 60, 60)),
    # Rook Ashen: Rook's grey shadow - ash hair, red visor, charcoal suit
    "BloodFalcon": dict(t="human", skin=(170, 160, 156), style="short", hair=(150, 150, 156), eyes="visor",
                        iris=(240, 40, 40), mouth="snarl", brows="angry", jaw=1.05, suit=(50, 50, 56),
                        collar=(200, 40, 40), bg=(30, 20, 22)),
    # Vito Brass: curly black hair, thick moustache-beard, warm grin, mustard jacket
    "AntonioGuster": dict(t="human", skin=(200, 140, 96), style="curly", hair=(30, 24, 20), beard=(40, 30, 24),
                          eyes="normal", iris=(80, 50, 30), mouth="grin", suit=(200, 160, 40), collar=(60, 40, 30),
                          bg=(50, 36, 16)),
    # Oru: long white hair, deep brown skin, round glasses, gentle smile, river-green robe
    "Baba": dict(t="human", skin=(110, 70, 50), style="long", hair=(236, 236, 240), eyes="glasses", mouth="smile",
                 suit=(40, 120, 90), collar=(230, 200, 120), bg=(16, 40, 34), old=True),
    # Grizzard: bear-like brawler (dog head template), grey-brown fur, light muzzle
    "Beastman": dict(t="dog", fur=(110, 96, 84), light=(210, 196, 170), suit=(150, 40, 40), bg=(36, 26, 22)),
    # Bongo: drummer droid (robot template), orange plating, blue eye bar
    "Billy": dict(t="robot", fur=(236, 150, 60), light=(250, 210, 150), eye=(60, 160, 250), suit=(60, 60, 70),
                  bg=(40, 26, 14)),
    # Quill: lilac alien with big amber eyes
    "BioRex": dict(t="alien", fur=(170, 140, 210), light=(220, 200, 240), eye=(250, 170, 40), suit=(60, 50, 90),
                   bg=(30, 24, 44)),
    # Lord Umbrage: dark chrome robot head, violet eye
    "BlackShadow": dict(t="robot", fur=(60, 62, 76), light=(130, 134, 150), eye=(170, 90, 250), suit=(28, 28, 36),
                        bg=(14, 12, 24)),
    # Prof. Bolt: bald, goggles pushed up, short grey beard, lab yellow
    "DrClash": dict(t="human", skin=(226, 176, 136), style="bald", goggles_up=(120, 200, 230), beard=(180, 180, 186),
                    eyes="normal", iris=(60, 80, 120), mouth="open", brows="white", suit=(240, 220, 90),
                    collar=(60, 60, 70), bg=(36, 36, 20), old=True),
    # Dr. Ilse Marrow: silver bob, glasses, headset, white coat with teal collar
    "DrStewart": dict(t="human", skin=(236, 196, 166), style="long", hair=(200, 200, 210), eyes="glasses",
                      headset=(90, 90, 100), mouth="lips", lips=(190, 90, 100), suit=(240, 240, 244),
                      collar=(40, 160, 170), bg=(24, 34, 44), jaw=0.9, lashes=True),
    # Zirk: spinning ball robot (ead template) in mint green
    "Draq": dict(t="ead", fur=(110, 220, 170), light=(200, 250, 220), suit=(40, 60, 70), bg=(14, 40, 34)),
    # Pip & Pop: a duo (twin template) in blue and yellow
    "GomarAndShioh": dict(t="twin", fur=(70, 120, 210), light=(250, 220, 90), suit=(240, 240, 240),
                          bg=(20, 30, 50)),
    # Jett Rowe: long blue hair, star-shaped earrings (none drawn), bright smile, white and coral suit
    "JackLevin": dict(t="human", skin=(240, 200, 170), style="long", hair=(60, 110, 230), eyes="normal",
                      iris=(40, 150, 140), mouth="smile", suit=(240, 240, 244), collar=(250, 110, 90),
                      bg=(30, 30, 50), jaw=0.9, lashes=True),
    # Hank Rivers: grey swept hair, short beard, narrow eyes, brown flight jacket
    "JamesMcCloud": dict(t="human", skin=(220, 170, 130), style="slick", hair=(150, 150, 156), beard=(130, 130, 136),
                         eyes="narrow", iris=(60, 90, 140), mouth="closed", suit=(120, 80, 50), collar=(220, 200, 160),
                         bg=(36, 30, 24), old=True),
    # Nova Kell: short curly red hair, goggles, freckled cheeks, cobalt suit
    "JodySummer": dict(t="human", skin=(240, 200, 170), style="curly", hair=(200, 70, 40), eyes="goggles",
                       iris=(60, 200, 230), mouth="smile", suit=(40, 60, 170), collar=(250, 200, 60),
                       bg=(20, 24, 50), jaw=0.92),
    "JodySummerAlt": dict(t="human", skin=(240, 200, 170), style="curly", hair=(200, 70, 40), eyes="normal",
                          iris=(60, 140, 90), mouth="grin", suit=(250, 200, 60), collar=(40, 60, 170),
                          bg=(40, 34, 14), jaw=0.92),
    # Ken Oda: spiky white hair, headset, determined brows, black and lime suit
    "JohnTanaka": dict(t="human", skin=(226, 180, 130), style="wild", hair=(240, 240, 240), headset=(60, 60, 70),
                       eyes="normal", iris=(40, 30, 30), mouth="closed", brows="angry", suit=(30, 30, 34),
                       collar=(160, 230, 60), bg=(20, 30, 14)),
    # Mira Sol: pink flat-top crop, lilac lips, sunny smile, gold suit
    "KateAlen": dict(t="human", skin=(170, 110, 80), style="flattop", hair=(250, 120, 190), eyes="normal",
                     iris=(80, 50, 30), mouth="lips", lips=(170, 100, 200), suit=(240, 190, 60), collar=(250, 250, 250),
                     bg=(50, 30, 40), jaw=0.88, lashes=True),
    # Zeph: sky-blue lizard pilot (lizard template)
    "Leon": dict(t="lizard", skin=(40, 50, 70), fur=(90, 170, 230), light=(200, 230, 250), suit=(240, 240, 240),
                 collar=(60, 120, 200), bg=(16, 30, 50)),
    # Rusty Cogs: wild orange hair, goggles, grin, oily blue overalls
    "MichaelChain": dict(t="human", skin=(230, 180, 140), style="wild", hair=(230, 120, 40), eyes="goggles",
                         iris=(250, 200, 80), mouth="grin", suit=(50, 80, 130), collar=(180, 120, 60),
                         bg=(30, 30, 40), stubble=True),
    # Circuit Hale: blue flat-top, headset, sharp jaw, silver suit
    "MightyGazelle": dict(t="human", skin=(210, 160, 120), style="flattop", hair=(60, 110, 220), headset=(200, 200, 210),
                          eyes="narrow", iris=(60, 200, 250), mouth="closed", suit=(170, 176, 190),
                          collar=(60, 110, 220), bg=(20, 26, 40), jaw=1.08),
    # Mr. Gizmo: purple alien tinkerer
    "MrEad": dict(t="alien", fur=(140, 90, 200), light=(200, 170, 240), eye=(120, 250, 200), suit=(240, 160, 40),
                  bg=(30, 20, 40)),
    # Lady Vector: straight black hair, teal visor, red lips, white and green suit
    "MrsArrow": dict(t="human", skin=(236, 190, 150), style="long", hair=(24, 24, 30), eyes="visor",
                     iris=(40, 220, 200), mouth="lips", lips=(200, 40, 60), suit=(240, 240, 244),
                     collar=(40, 170, 90), bg=(20, 40, 30), jaw=0.9),
    # Pell Corrin: dark hair in a short crop, round goggles, open grin, rust suit
    "Octoman": dict(t="human", skin=(150, 100, 74), style="short", hair=(40, 30, 26), eyes="goggles",
                    iris=(240, 160, 60), mouth="grin", suit=(170, 80, 50), collar=(240, 220, 180), bg=(40, 24, 18)),
    # Skeff: small orange droid (robot template), green eye bar
    "Pico": dict(t="robot", fur=(220, 110, 40), light=(250, 190, 120), eye=(80, 240, 120), suit=(50, 50, 56),
                 bg=(36, 20, 10)),
    # Duke Rally: black slick hair, shades, moustache-beard, red showman suit
    "RogerBuster": dict(t="human", skin=(226, 170, 124), style="slick", hair=(20, 20, 26), beard=(30, 28, 30),
                        eyes="shades", mouth="grin", suit=(190, 30, 40), collar=(250, 220, 90), bg=(40, 16, 20),
                        jaw=1.1),
    # Taro Brand: bald, full dark beard, cap, kind narrow eyes, olive suit
    "SamuraiGoroh": dict(t="human", skin=(214, 160, 110), style="cap", hair=(200, 80, 40), beard=(50, 36, 28),
                         eyes="narrow", iris=(40, 30, 20), mouth="smile", suit=(90, 110, 50), collar=(240, 230, 200),
                         bg=(30, 34, 16)),
    "SamuraiGorohAlt": dict(t="human", skin=(214, 160, 110), style="bald", beard=(50, 36, 28), eyes="shades",
                            mouth="grin", suit=(240, 230, 200), collar=(90, 110, 50), bg=(34, 30, 20)),
    # Old Grayson: long grey hair, glasses, lips, lavender suit
    "SilverNeelsen": dict(t="human", skin=(236, 196, 170), style="long", hair=(176, 176, 184), eyes="glasses",
                          mouth="lips", lips=(180, 100, 120), suit=(150, 120, 190), collar=(240, 240, 244),
                          bg=(30, 26, 40), old=True, jaw=0.9),
    # Captain Vector: green helmet with silver stripes, no visor, broad grin
    "SuperArrow": dict(t="human", scale=0.9, skin=(220, 166, 120), helmet=dict(col=(40, 150, 80), emblem="stripes",
                       ecol=(220, 224, 230)), eyes="normal", iris=(50, 80, 140), mouth="grin", jaw=1.1,
                       suit=(240, 240, 244), collar=(40, 150, 80), bg=(16, 40, 24)),
    # Hollow King: pale grey alien with a dark eye glow
    "TheSkull": dict(t="alien", fur=(200, 200, 196), light=(240, 240, 236), eye=(60, 60, 90), suit=(90, 20, 60),
                     bg=(24, 18, 26)),
    # Vex Kragg: crimson lizard rogue
    "Zoda": dict(t="lizard", skin=(30, 20, 24), fur=(190, 50, 60), light=(250, 150, 140), suit=(30, 30, 40),
                 collar=(250, 200, 60), bg=(40, 14, 18)),
    # Max Volume: race-control host on a green monitor
    "MrZero": dict(t="zero", skin=(110, 200, 140), hair=(20, 70, 40), eye=(255, 200, 60), suit=(30, 90, 60),
                   bg=(20, 90, 50)),
    # pit-lane host in the credits: copper hair, mint visor, grey and orange suit
    "MenuLady": dict(t="human", skin=(230, 186, 150), style="curly", hair=(190, 100, 50), eyes="visor",
                     iris=(120, 250, 210), mouth="smile", suit=(80, 84, 90), collar=(250, 140, 50), bg=(24, 24, 28)),
}
