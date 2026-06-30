import json

inp = "/home/ids/diecidue/data/captions/full_chunks/chunk_088.json"
out = "/home/ids/diecidue/data/captions/full_chunks/chunk_088.jsonl"

with open(inp) as f:
    items = json.load(f)

# caption keyed by (dataset, image_id)
caps = {
    ("aerialwaste_m2", "5971"): "Aerial view of agricultural land: a tan/reddish ploughed field upper-left, large green grassy/crop fields centre and right, bordered by dark green tree lines and hedgerows. No waste piles or dumped objects visible.\n  Labels: none",
    ("aerialwaste_m2", "578"): "Aerial view of an industrial site: rows of long green-roofed warehouses on the left, a paved yard with a few small structures and parked vehicles, a straight road lined with trees, and green fields beyond. Bulky machinery and scattered objects sit in the yard near centre; several rectangular bins/containers line the buildings.\n  Labels: Bulky items, Containers",
    ("aerialwaste_m2", "2012"): "Aerial view of a scrap/storage yard: an open paved lot strewn with heaped grey and metallic debris piles centre-left, a large light-roofed warehouse on the right, and a railway corridor along the bottom. Heaped scrap and bulky equipment fill the lot (bulky items, rubble, unidentified material); rectangular bins/containers stand near the buildings.\n  Labels: Bulky items, Containers, Rubble, Unknown material",
    ("aerialwaste_m2", "5902"): "Aerial view of flat agricultural fields, finely furrowed green and tan cropland divided by faint tracks, with a thin tree line and a small dark square structure lower-centre. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "1794"): "Low-altitude oblique drone shot of dry scrubby hillside terrain, pale tan grass and bare earth with sparse green-grey vegetation and rock; deep shadow across the lower area. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "633"): "Drone view of a dusty bare-earth lot with patches of dried grass; a fenced area, scattered greenery and a small square concrete pit/structure lower-centre. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "4398"): "Close drone view straight down onto a green field with fine diagonal crop/till rows in striped pale-and-green texture. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "2965"): "Close drone view of a dense green crop field, uniform rows of leafy plants filling the frame in a regular vertical texture. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "4328"): "Drone view of sparse woodland/scrub: bare brownish ground with thin tree trunks casting diagonal shadows and patches of low green growth. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "1907"): "Oblique drone view of a pale gravel/dirt track running diagonally, bordered by dry scrubby vegetation and scattered light-coloured debris/litter along the verges (blurred). No clearly defined waste pile labeled.\n  Labels: none",
    ("dronewaste_paper10", "457"): "Close drone view of a green crop field with fine vertical rows and pale gaps, slightly motion-blurred uniform vegetation texture. No waste piles or dumped objects visible.\n  Labels: none",
    ("aerialwaste_m2", "1367"): "Aerial view of a green park-like clearing ringed by trees, with buildings and a parking area lower-right and a road network. A heaped grey/white pile of rubble and bulky debris sits in the clearing left of centre, distinguished by its irregular angular mound against the grass.\n  Labels: Bulky items, Rubble",
    ("aerialwaste_m2", "11594"): "Aerial view of large dark-green crop fields meeting a brown scrub strip; a multi-lane road with vehicles runs across the bottom, joined by a side track. No waste piles or dumped objects visible.\n  Labels: none",
    ("aerialwaste_m2", "4339"): "Aerial view of rectangular farm fields in greens and tan-brown, divided by tree lines and hedgerows with a faint track at right. No waste piles or dumped objects visible.\n  Labels: none",
    ("aerialwaste_m2", "9291"): "Aerial view of a dense residential neighbourhood: rows of houses with red/brown roofs, gardens, tree-lined streets and parked cars; a small swimming pool lower-left. No waste piles or dumped objects visible.\n  Labels: none",
    ("aerialwaste_m2", "2231"): "Aerial view of a rural property cluster: houses and sheds with a central yard, gardens and field plots around. The yard centre-left holds parked vehicles, scattered bulky equipment and stacked materials, with rectangular bins/containers and indistinct heaped material amid the structures.\n  Labels: Bulky items, Containers, Unknown material",
    ("dronewaste_paper10", "4623"): "Close drone view of a flat pale yellow-green grassy surface with faint mottled texture, nearly uniform. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "1152"): "Oblique drone view of dry grassland on a slope, tan and green patchy vegetation with bare-earth scars and tree shadows; a cleared earthy patch lower-centre. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "4614"): "Drone view of a pale rubble-strewn slope: light grey/white broken material and dusty soil with sparse green growth; two distinct white rectangular blocks (cut stone/slabs) sit lower-right. The scattered pale broken stone and slabs across the centre-right are construction and demolition material.\n  Labels: Construction and demolition materials",
    ("aerialwaste_m2", "1222"): "Aerial view of a semi-rural estate: buildings with a courtyard and tree-lined drive upper-centre, surrounded by woodland, scrub, olive groves and grassy clearings; spiral terraced earthworks lower-right. Scattered light-coloured debris and heaped material with bulky objects sit in a clearing near centre by the track junction.\n  Labels: Bulky items, Rubble, Unknown material",
    ("aerialwaste_m2", "4265"): "Aerial view of an affluent residential area amid woodland: villas with red roofs, several blue swimming pools, lawns, hedges and winding roads, with dense green tree cover. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "3209"): "Drone view of a dry brushy slope with leafless shrubs and bare brown earth; pale rubble and light-coloured debris cluster lower-left with a hint of blue and white material. Broken pale construction rubble fills the lower-left clearing, and bright plastic packaging items are mixed among it.\n  Labels: Construction and demolition materials, Plastic packaging",
    ("dronewaste_paper10", "392"): "Close drone view of dense green tree canopy/scrub, mottled foliage of varied greens filling the frame. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "3701"): "Close drone view straight down onto dense yellow-green low vegetation/crop with fine mottled texture and faint diagonal shadow lines. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "3777"): "Drone view of a field divided diagonally between dense yellowish standing crop and shorter green growth, fine textured vegetation. No waste piles or dumped objects visible.\n  Labels: none",
    ("aerialwaste_m2", "175"): "Aerial view of farmland: a large tan harvested/ploughed field above a road, green meadows and tree clusters below. A bright white and tan heaped pile of rubble and debris sits prominently in the centre meadow; smaller scattered objects lie in the grassy plot at left.\n  Labels: Bulky items, Rubble, Unknown material",
    ("aerialwaste_m2", "2185"): "Aerial view of a horticultural/nursery site: long greenhouse-like rows and plots with surrounding buildings, vehicles and a yard. Scattered bulky items, stacked materials, bins/containers, pale plastic sheeting and indistinct heaped material occupy the yard areas among the structures.\n  Labels: Bulky items, Containers, Plastic, Rubble, Unknown material",
    ("aerialwaste_m2", "7071"): "Aerial view of dense dark-green forest canopy covering most of the frame, with a small building and clearing at the lower-right edge. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "3315"): "Close oblique drone view of a green-and-tan field with fine horizontal furrow/crop rows, partly shadowed lower edge. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "3693"): "Low oblique drone view of a building with a tiled red-brown pitched roof and an adjoining darker fibre-cement roof; debris and broken material lie in the yard at left. The grey corrugated fibre-cement roofing sheets are asbestos; the broken rubble and structural debris around the building are construction and demolition material.\n  Labels: Asbestos, Construction and demolition materials",
    ("aerialwaste_m2", "7536"): "Aerial view of a patchwork of green crop fields of varying tones, divided by hedgerows and tracks, bordered by a muddy river/canal at lower-left with riverbank trees. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "641"): "Drone view of green field/scrub with a reddish-brown bare earth strip, partly obscured by dark blotchy image artefacts/shadows. No waste piles or dumped objects visible.\n  Labels: none",
    ("aerialwaste_m2", "3530"): "Aerial view of large dark-green crop fields divided by faint straight tracks and a road along the bottom, uniform agricultural land. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "1789"): "Drone view of a sandy/dusty bare-earth bank marked with tyre tracks meeting a brown muddy water pool at the bottom; sparse debris and twigs scattered on the soil. No clearly defined waste pile labeled.\n  Labels: none",
    ("dronewaste_paper10", "2646"): "Drone view of a rubble-strewn ground: pale grey broken stone and dusty soil with a patch of low green vegetation upper-centre. No waste pile labeled.\n  Labels: none",
    ("aerialwaste_m2", "11390"): "Aerial view of a multi-lane highway with vehicles running diagonally, a large light-roofed industrial building and a busy parking lot full of cars on the right, green field at left. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "4114"): "Close drone view of yellow-green grassy field with a dark green vegetated drainage line/furrow crossing diagonally. No waste piles or dumped objects visible.\n  Labels: none",
    ("aerialwaste_m2", "4825"): "Aerial view of a hillside vineyard: regular dark rows of vines on terraced slopes, bordered by woodland, with a farm building and small yard at centre-bottom and access tracks. No waste piles or dumped objects visible.\n  Labels: none",
    ("aerialwaste_m2", "1413"): "Aerial view of a rural farmstead: long pink/red-roofed barns and sheds around a large pale gravelled yard, surrounded by green fields. Some dark vehicles/equipment sit in the yard but no defined waste pile is visible.\n  Labels: none",
    ("dronewaste_paper10", "2978"): "Drone view of a dry tilled field strip with sparse green weeds and a wire fence line bordering denser green vegetation at right; a person stands at the top edge. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "411"): "Close drone view of dense blue-green shrub foliage filling most of the frame, with a strip of paler grass at upper-left. No waste piles or dumped objects visible.\n  Labels: none",
    ("aerialwaste_m2", "4845"): "Aerial view of a hillside vineyard estate: long rows of vines on sloping terraces, a central farmhouse with red roofs, access tracks and surrounding woodland. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "2940"): "Drone view split between a lush green crop field at left and dry tan grass/bare soil at right, divided by a vegetation line. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "2701"): "Close oblique drone view of a worn concrete/paved surface with moss and grass growing through cracks, mottled green-and-grey. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "3927"): "Close drone view of a dense yellow-green crop field with fine diagonal texture and faint reddish patches. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "2568"): "Drone view of a paved/gravel surface with faint grid markings at right and a strip of green grass with a tree/bush at left. No waste piles or dumped objects visible.\n  Labels: none",
    ("aerialwaste_m2", "11083"): "Aerial view of a dense industrial estate: large warehouses with metal roofs, paved yards, internal roads and vehicles. A central open storage yard is filled with heaped scrap, stacked materials, bins/containers and bulky equipment in mixed colours, with indistinct piled material throughout.\n  Labels: Bulky items, Containers, Rubble, Unknown material",
    ("aerialwaste_m2", "2325"): "Aerial view of an abandoned/cleared site: extensive pale grey concrete slabs and foundation remains amid encroaching trees and scrub, with houses at the upper-right edge. The grey broken slab field and scattered rubble across the centre are demolition debris with bulky fragments and indistinct heaped material.\n  Labels: Bulky items, Rubble, Unknown material",
    ("dronewaste_paper10", "1701"): "Oblique drone view of a scrubby slope, dry tan and rusty-brown vegetation with bare patches; a cluster of pale white/grey scattered fragments and bulky debris lies centre. The heterogeneous heap of light-coloured scattered refuse centre-frame is mixed dumped items.\n  Labels: Mixed items",
    ("dronewaste_paper10", "4137"): "Close drone view of a green grassy field with a faint curving dirt track and darker vegetation lines at lower-right. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "3487"): "Close drone view of a yellow-green crop/scrub field crossed by a darker green vegetated band/ditch on the right. No waste piles or dumped objects visible.\n  Labels: none",
    ("aerialwaste_m2", "2469"): "Aerial view of a rural property with houses, a swimming pool and sheds set among fields; a large pale tan bare-earth/dirt lot dominates the left. Stacked materials, bulky equipment, pale plastic-covered piles and indistinct heaped rubble cluster in the yards around the central buildings.\n  Labels: Bulky items, Plastic, Rubble, Unknown material",
    ("dronewaste_paper10", "661"): "Close oblique drone view of overlapping pitched roofs covered in reddish-brown patterned tiles, with shadowed gaps. No waste piles or dumped objects visible.\n  Labels: none",
    ("aerialwaste_m2", "2760"): "Aerial view of large dark-green crop fields with fine vertical striping, divided by faint pale boundary lines, and a small dark scorched patch at lower-left. No waste piles or dumped objects visible.\n  Labels: none",
    ("aerialwaste_m2", "332"): "Aerial view of dense dark blue-green forest canopy with several pale tan bare-earth clearings near the top-centre linked by a faint track. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "693"): "Drone view with heavy dark image artefacts/occlusions; visible patches show pale soil, green vegetation and muddy ground in a fragmented, blurred scene. No waste pile clearly identifiable.\n  Labels: none",
    ("dronewaste_paper10", "2111"): "Drone view straight down onto a grey concrete surface (top) meeting a row of green and tan rectangular containers/skips at the bottom. The colourful rectangular bins are containers, but GT is empty so none labeled.\n  Labels: none",
    ("aerialwaste_m2", "5668"): "Aerial view of dark-green crop fields divided by a curving pale track and a hedgerow with a small bushy tree at centre. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "2458"): "Close drone view straight down onto stacked corrugated grey roofing sheets with regular fine ribbing, bordered by darker grey debris/ground at the edges. The ribbed grey fibre-cement sheets filling the frame are asbestos.\n  Labels: Asbestos",
    ("aerialwaste_m2", "2954"): "Aerial view of an old village/farm cluster: long terraced buildings and barns with weathered roofs, a winding road and tree-lined fields; cluttered yards with debris. Yards hold bins/containers, pale plastic sheeting, scattered rubble and indistinct heaped material among the buildings and the bare lot upper-right.\n  Labels: Containers, Plastic, Rubble, Unknown material",
    ("dronewaste_paper10", "597"): "Drone view straight down onto a reddish-brown paved/asphalt path running vertically, bordered by green grass and weedy vegetation. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "2476"): "Drone view straight down onto two long grey corrugated/ribbed roofing sheets with a green grassy strip at left. The ribbed grey fibre-cement roof sheets are asbestos.\n  Labels: Asbestos",
    ("dronewaste_paper10", "1248"): "Drone view of a pale sandy/gravel track running diagonally through dry scrubby grassland with patches of green vegetation. No waste piles or dumped objects visible.\n  Labels: none",
    ("aerialwaste_m2", "11329"): "Aerial view of a large tan harvested/ploughed field with faint till rows, bordered by tree lines and a green strip at top, darker fields at the corners. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "160"): "Oblique drone view of a grassy green slope with bare-earth scars, pale rocky patches lower-left and hazy over-exposed terrain at top. No waste piles or dumped objects visible.\n  Labels: none",
    ("aerialwaste_m2", "5575"): "Aerial view of dense mottled forest canopy, dark green trees interspersed with paler grey-green bare patches and gaps, covering the whole frame. No waste piles or dumped objects visible.\n  Labels: none",
    ("aerialwaste_m2", "6231"): "Aerial view of a large green grassy field with a pale industrial building and small parking area at the upper-left, bordered by hedgerows and tracks. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "2686"): "Drone view of a bare-earth clearing with patches of green grass and scattered light-coloured debris; dark bags and bright pale items lie centre. The scattered white/light plastic bags and packaging items across the clearing centre are plastic packaging.\n  Labels: Plastic packaging",
    ("dronewaste_paper10", "4340"): "Close drone view straight down onto a pale yellow-green ribbed surface (likely tarpaulin/netted ground) with a dark irregular gap/hole at upper-right and small dark debris specks. No waste pile labeled.\n  Labels: none",
    ("dronewaste_paper10", "2168"): "Drone view straight down onto a grey concrete/asphalt yard with faint painted square markings and a white car at the lower-right. No waste piles or dumped objects visible.\n  Labels: none",
    ("aerialwaste_m2", "1478"): "Aerial view of farmland: large tan and green fields divided by a straight road, with a small farm building cluster and yard at centre. Bulky equipment, pale plastic sheeting/wrapped bales and indistinct stored material sit in the yard beside the buildings.\n  Labels: Bulky items, Plastic, Unknown material",
    ("dronewaste_paper10", "59"): "Close drone view of dense blue-green shrub foliage with a paler green grassy patch at upper-left, mottled vegetation texture. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "730"): "Close drone view straight down onto dry mottled grassland, tan and brown patchy turf with sparse green tufts. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "4582"): "Close drone view of a green field with fine diagonal mowing/till rows and pale cut-vegetation streaks. No waste piles or dumped objects visible.\n  Labels: none",
    ("aerialwaste_m2", "11175"): "Aerial view of a large pale tan ploughed/bare field with subtle tonal banding and a thin dark hedgerow line crossing diagonally at lower-left. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "1528"): "Oblique drone view of a dry scrubby slope, tan grass and bare earth with patchy green low vegetation and scattered pale stones/rock. No waste pile clearly identifiable.\n  Labels: none",
    ("dronewaste_paper10", "841"): "Oblique drone view of large blue solar/photovoltaic panel arrays on a building roof, with a grey corrugated roof and structures at left. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "4163"): "Drone view of grassland heavily occluded by large black image artefacts/shadows; visible patches show pale tan grass with sparse blue/green specks. No waste pile clearly identifiable.\n  Labels: none",
    ("aerialwaste_m2", "8309"): "Aerial view of farmland boundary: a large pale tan ploughed field on the left meeting dense dark-green woodland on the right along a diagonal tree line. No waste piles or dumped objects visible.\n  Labels: none",
    ("aerialwaste_m2", "9113"): "Aerial view of a hillside settlement: houses with red roofs, a blue swimming pool, long greenhouse tunnels and terraced olive groves around a winding road; a flat-roofed building cluster at centre. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "4374"): "Close drone view of a green field with fine diagonal crop/till rows and pale mottled patches, uniform vegetation texture. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "3381"): "Close drone view straight down onto dense yellow-green low vegetation with a faint green diagonal streak, near-uniform texture. No waste piles or dumped objects visible.\n  Labels: none",
    ("aerialwaste_m2", "1913"): "Aerial view of an industrial estate: rows of large grey-roofed warehouses with paved yards, internal lanes and parked vehicles/trucks. Yards between buildings hold stacked materials, bins/containers, pale plastic-wrapped goods, bulky equipment and indistinct heaped material.\n  Labels: Bulky items, Containers, Plastic, Rubble, Unknown material",
    ("dronewaste_paper10", "3496"): "Close drone view straight down onto a uniform yellow-green low crop/grass surface with faint mottled and diagonal texture. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "4666"): "Close drone view of a green grassy slope with pale grey diagonal striping/wear bands and scattered light specks. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "4511"): "Close drone view of a field with fine diagonal crop/mow rows in alternating green and tan stripes. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "2582"): "Drone view of a grey paved/cobbled area with a green grass strip and small bushes at left, a zebra-crossing and a small figure; large black artefact occludes the lower-right corner. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "1770"): "Oblique drone view of pale sandy bare-earth ground crossed by faint tyre tracks, with sparse dry scrub at the edges. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "1909"): "Drone view of dry scrubby terrain, mottled tan and green low vegetation, partly occluded by black image artefacts at the bottom. No waste piles or dumped objects visible.\n  Labels: none",
    ("aerialwaste_m2", "11597"): "Aerial view of farmland: green meadows and a large tan ploughed field divided by hedgerows and a road, with scattered trees and a small structure/clearing at centre-right. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "2226"): "Close low oblique drone view of dark debris and crumpled fabric/material against pale grey rubble and concrete; bundled dark and tan cloth lies centre. The crumpled dark fabric bundles amid the debris are textile waste.\n  Labels: Textile",
    ("dronewaste_paper10", "885"): "",  # placeholder, will not be used
    ("dronewaste_paper10", "site6_885"): "",
    ("dronewaste_paper10", "4956"): "Drone view of dense leafless brown scrub/woodland, bare branches over brownish ground with a faint diagonal track. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "130"): "Close drone view of a pale concrete/dusty surface meeting a strip of green grass at the bottom; faint scuff marks and a small light object upper-left. No waste piles or dumped objects visible.\n  Labels: none",
    ("aerialwaste_m2", "10223"): "Aerial view of farmland edge: a large pale tan ploughed field on the left meeting dense dark-green woodland on the right along a curving boundary. No waste piles or dumped objects visible.\n  Labels: none",
    ("aerialwaste_m2", "41"): "Aerial view of an industrial complex: a large blue solar-panelled roof and long grey warehouses with paved yards and parked vehicles, bordered by green fields. Rectangular bins/containers and a few stored units sit in the yards beside the buildings.\n  Labels: Containers",
    ("aerialwaste_m2", "6057"): "Aerial view of an industrial site: a large dark solar/PV roof, long warehouses and paved yards with scattered vehicles, surrounded by green fields and tracks. No defined waste pile visible.\n  Labels: none",
    ("dronewaste_paper10", "2622"): "Aerial view of a wooded valley: brown leafless trees on steep slopes, green meadow strips, and a winding track/stream line at centre. No waste piles or dumped objects visible.\n  Labels: none",
    ("dronewaste_paper10", "4719"): "Drone view of dry scrubby ground with pale sandy patches, dead palm-frond/brush debris and sparse green tufts. No defined waste pile labeled.\n  Labels: none",
    ("aerialwaste_m2", "2965"): "Aerial view of a rural farmstead: buildings with grey and red roofs and a courtyard, surrounded by green crop fields, a silo and access tracks. A pale grey heap of rubble and indistinct piled material sits in the bare yard beside the buildings at centre.\n  Labels: Rubble, Unknown material",
}

# fix site6_885 caption
caps[("dronewaste_paper10", "site6_885")] = "Oblique drone view of a scrubby slope, dry rusty-brown and green low vegetation with bare patches; scattered pale grey debris and a white object cluster lower-left. No defined waste pile labeled per GT.\n  Labels: none"

# map image_id 4956 caption to proper text (it's site16_991)
caps[("dronewaste_paper10", "4956")] = "Drone view of dense leafless brown scrub/woodland, a tangle of bare branches over brownish ground crossed by faint diagonal track lines. No waste piles or dumped objects visible.\n  Labels: none"

# image_id 1714 -> site6_885
caps[("dronewaste_paper10", "1714")] = caps[("dronewaste_paper10", "site6_885")]

written = 0
missing = []
with open(out, "w") as fo:
    for it in items:
        key = (it["dataset"], it["image_id"])
        cap = caps.get(key)
        if cap is None or cap == "":
            missing.append(key)
            continue
        rec = {
            "dataset": it["dataset"],
            "image_id": it["image_id"],
            "image_path": it["image_path"],
            "gt_categories": it["gt_categories"],
            "caption": cap,
            "model": "claude-code-agent",
        }
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
        written += 1

print("written", written)
print("missing", missing)
