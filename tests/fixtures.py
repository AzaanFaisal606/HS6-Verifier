"""Raw pdftotext -layout line blocks lifted verbatim from PTC-2025-26.pdf.

Column spacing is intentional and must not be reflowed: the parser keys
specific-duty detection on absolute indent. Dash counts encode label nesting.
"""

FIX_4202 = """\
42.02       Trunks, suit- cases, vanity- cases, executive- cases,
            briefcases, school satchels, spectacle cases, binocular
            cases, camera cases, musical instrument cases, gun
            cases, holsters and similar containers; travelling- bags,
            insulated food or beverages bags, toilet bags,
            rucksacks, handbags, shopping- bags, wallets, purses,
            map- cases, cigarette- cases, tobacco- pouches, tool
            bags, sports bags, bottle- cases, jewellery boxes,
            powder- boxes, cutlery cases and similar containers, of
            leather or of composition leather, of sheeting of
            plastics, of textile materials, of vulcanised fibre or of
            paperboard, or wholly or mainly covered with such
            materials or with paper.

            - Trunks, suit- cases, vanity- cases, executive- cases, brief
            cases, school satchels and similar containers:
            - - With outer surface of leather or of composition leather:

4202.1120   - - - Suit-cases, of leather or composition leather             20
4202.1190   - - - Other                                                     20
            - - With outer surface of plastics or of textile materials:

4202.1210   - - - Travelling bags of plastics or textile materials          20
4202.1220   - - - Suit cases of plastics or textile materials               20
4202.1290   - - - Other                                                     20
4202.1900   - - Other                                                       20
            - Handbags, whether or not with shoulder strap, including
            those without handle:
4202.2100   - - With outer surface of leather or of composition leather     20

4202.2200   - - With outer surface of sheeting of plastics or of textile    20
            materials
4202.2900   - - Other                                                       20
            - Articles of a kind normally carried in the pocket or in the
            handbag:
4202.3100   - - With outer surface of leather or of composition leather     20

4202.3200   - - With outer surface of sheeting of plastics or of textile    20
            materials
4202.3900   - - Other                                                       20
            - Other:
4202.9100   - - With outer surface of leather or of composition leather     20

4202.9200   - - With outer surface of sheeting of plastics or of textile    20
            materials
4202.9900   - - Other                                                       20
"""

FIX_2501 = """\
25.01       Salt (including table salt and denatured salt) and pure
            sodium chloride, whether or not in aqueous solution or
            containing added anti- caking or free- flowing agents;
            sea water.
2501.0010   - - - Table salt                                           20
            - - - Rock salt:
2501.0021   - - - - Pink rock salt                                     20
2501.0029   - - - - Other                                              20
2501.0030   - - - Sea salt                                             20
2501.0090   - - - Other                                                20
"""

FIX_8544 = """\
85.44       Insulated wire, cable and other insulated electric
            conductors.
8544.2000   - Co- axial cable and other co- axial electric conductors        20

            - Ignition wiring sets and other wiring sets of a kind used in
            vehicles, aircraft or ships:
            - - - Of a kind used in vehicles of chapter 87:
8544.3011   - - - - Wiring sets and cable sets for vehicles of heading       35
            87.03 and vehicles of sub-headings 8704.2190, 8704.3130
8544.3019   - - - - Other                                                    35
8544.3090   - - - Other                                                      20
"""

FIX_4402 = """\
44.02        Wood charcoal (including shell or nut charcoal),
            whether or not agglomerated.
4402.1000    - Of bamboo                                                      0
4402.2000   - Of shell or nut                                                 0
4402.9000    - Other                                                          0
"""

FIX_0205 = """\
0205.0000   Meat of horses, asses, mules or hinnies, fresh, chilled    20
"""

FIX_0101 = """\
01.01       Live horses, asses, mules and hinnies.
            - Horses:
0101.2100   - - Pure-bred breeding animals                                0
0101.2900   - - Other                                                     0
0101.3000   - Asses                                                       0
"""
