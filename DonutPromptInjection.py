"""
DonutPromptInjection - A comprehensive style prompt injector for ComfyUI
Allows combining style prompts with user prompts in either order.
Chainable for combining multiple styles.
"""
import random

# =============================================================================
# STYLE DEFINITIONS
# =============================================================================

PHOTO_STYLES = {
    "None": "",
    # Vintage & Film Stocks
    "Vintage Kodak": "vintage Kodak Portra 400 film photograph, creamy skin tones, soft grain, natural color rendition",
    "Retro Polaroid": "instant Polaroid photograph, faded colors, white border frame, slight color shift, nostalgic lo-fi aesthetic, soft focus",
    "70s Faded Film": "1970s film photograph, faded tones, visible grain, soft dreamy glow",
    "Cinematic Kodachrome": "Kodachrome slide film, rich saturated colors, deep blacks, vintage cinema look",
    "Grainy Tri-X": "Kodak Tri-X 400 black and white film, high contrast, prominent grain texture, deep rich blacks, classic photojournalism look",
    "Cross-Processed": "cross-processed slide film, unusual color shifts, high contrast, cyan shadows, yellow highlights, experimental vintage look",
    "Expired Film": "shot on expired film stock, unpredictable color shifts, light leaks, heavy grain, faded pastel tones, dreamy imperfections",
    # Black & White
    "Classic Noir B&W": "classic film noir black and white, deep shadows, stark contrast, moody atmospheric",
    "Soft B&W Portrait": "fine art black and white portrait, smooth tonal gradations, creamy skin tones, elegant timeless",
    "Gritty Documentary": "gritty documentary black and white, high contrast, harsh shadows, raw unfiltered realism, street photography aesthetic",
    # Modern Digital Looks
    "Clean Commercial": "professional commercial photography, sharp focus, high production value, polished finish",
    "Moody Editorial": "editorial fashion photography, sophisticated atmosphere, magazine quality, deliberate aesthetic",
    "Instagram Lifestyle": "lifestyle photography, clean minimal aesthetic, social media polished, aspirational feel",
    "Dark & Moody": "dark moody photography, deep shadows, rich blacks, dramatic atmosphere",
    "Light & Airy": "light airy photograph, soft pastels, dreamy ethereal quality, gentle feel",
    # Specialized Genres
    "Vintage Sports Action": "vintage sports photography, slight motion blur, grainy film texture, dynamic action frozen, nostalgic athletic",
    "Food Magazine": "professional food photography, appetizing presentation, editorial quality, delicious styling",
    "Real Estate HDR": "real estate photography, clean architectural, HDR balanced exposure, spacious feel",
    "Concert Photography": "live concert photography, motion blur, high ISO grain, energetic atmosphere, live performance feel",
    "Underwater Dreamy": "underwater photography, floating ethereal quality, submerged perspective, aquatic distortion",
    "Infrared Surreal": "infrared photography, white foliage, dark skies, surreal dreamlike, false color, otherworldly landscape",
}

ILLUSTRATION_STYLES = {
    "None": "",
    # Comics & Graphic Novels
    "Classic Comic Book": "bold comic book illustration, heavy black ink outlines, cel shading, vibrant primary colors, halftone dot patterns, dynamic composition",
    "Modern Graphic Novel": "graphic novel illustration, refined linework, sophisticated color palette, cinematic panel composition, crosshatching shadows",
    "Underground Comix": "underground comics art, raw scratchy linework, high contrast black and white, rebellious aesthetic, hand-drawn imperfections, bold stark shadows",
    "European Bande Dessinee": "Franco-Belgian comic art, clean precise ligne claire outlines, flat vibrant colors, detailed backgrounds, elegant composition, minimal shading",
    "Superhero Bronze Age": "bronze age superhero comics, dynamic poses, rich saturated colors, detailed musculature, dramatic foreshortening, bold speed lines",
    # Manga & Anime
    "Classic Manga": "detailed manga illustration, clean precise linework, screentone shading, expressive faces, dynamic action lines, high contrast black and white",
    "Soft Anime": "soft anime illustration, large expressive eyes, pastel color palette, gentle gradients, luminous highlights, delicate linework, dreamy atmosphere",
    "Shonen Action": "shonen manga style, intense dynamic poses, bold speed lines, dramatic angles, spiky detailed hair, high energy composition, strong contrast",
    "Shoujo Romance": "shoujo manga style, sparkling eyes, flowing hair, floral decorative elements, soft screentones, dreamy backgrounds, elegant thin linework",
    "Mecha Anime": "mecha anime illustration, precise mechanical details, metallic reflections, technical linework, dynamic perspective, chrome highlights",
    # Children's Book
    "Storybook Whimsy": "whimsical children's book illustration, soft rounded shapes, warm cheerful colors, gentle textures, friendly inviting style, playful composition",
    "Modern Picture Book": "contemporary picture book art, bold simplified shapes, vibrant flat colors, charming character design, clean digital finish, playful geometry",
    "Vintage Golden Book": "vintage golden book illustration, mid-century aesthetic, limited warm color palette, textured paper look, nostalgic charm, soft painted edges",
    "Scandinavian Children's": "Scandinavian children's illustration, minimalist clean design, muted earthy palette, simple geometric shapes, cozy hygge atmosphere, gentle linework",
    # Traditional Media
    "Delicate Watercolor": "delicate watercolor painting, soft transparent washes, visible brush strokes, muted pastel palette, wet-on-wet blending, paper texture visible",
    "Rich Gouache": "gouache illustration, opaque matte finish, bold flat color areas, visible brushwork, rich saturated hues, slightly chalky texture",
    "Ink Wash Sumi-e": "sumi-e ink wash painting, fluid expressive brushstrokes, elegant minimalism, subtle tonal gradations, zen aesthetic, empty space emphasis",
    "Colored Pencil": "colored pencil illustration, layered waxy strokes, soft blended gradients, visible paper grain, rich burnished colors, delicate hatching",
    "Oil Pastel": "oil pastel illustration, rich creamy texture, bold expressive strokes, vibrant saturated colors, soft blended edges, visible grain",
    "Scratchboard": "scratchboard illustration, fine white lines on black, intricate crosshatching, dramatic high contrast, engraving-like detail, Victorian aesthetic",
    # Editorial & Concept
    "Editorial Magazine": "editorial illustration, conceptual symbolic imagery, sophisticated color palette, thoughtful composition, contemporary graphic design influence, bold shapes",
    "Concept Art Painterly": "painterly concept art, loose expressive brushwork, atmospheric depth, rich color harmony, artistic visible strokes",
    "Technical Illustration": "technical illustration style, precise clean linework, cutaway details, instructional clarity, subtle flat shading, diagram aesthetic",
    "Fashion Illustration": "fashion illustration, elongated elegant proportions, fluid gestural lines, sophisticated minimal palette, artistic loose rendering, expressive movement",
    # Stylized & Decorative
    "Art Nouveau Illustration": "art nouveau illustration, flowing organic curves, decorative floral borders, elegant feminine forms, muted jewel tones, ornamental linework",
    "Art Deco Glamour": "art deco illustration, geometric angular forms, luxurious metallic accents, bold symmetrical composition, sophisticated limited palette, elegant stylization",
    "Retro Mid-Century": "mid-century modern illustration, atomic age aesthetic, limited color palette, geometric simplified shapes, vintage print texture, bold graphic design",
    "Psychedelic": "psychedelic illustration, swirling organic patterns, vibrant clashing colors, optical distortions, intricate decorative details, trippy flowing forms",
    "Flat Vector": "flat vector illustration, clean geometric shapes, bold limited color palette, crisp hard edges, minimal gradients, contemporary graphic style",
    "Linocut Print": "linocut print illustration, bold carved lines, visible woodgrain texture, limited two-tone palette, folk art aesthetic, handmade imperfections",
}

DIGITAL_3D_STYLES = {
    "None": "",
    # Animated Film Styles
    "Pixar Style 3D": "Pixar-style 3D render, smooth subsurface scattering, soft ambient occlusion, appealing character design, vibrant colors",
    "Dreamworks Animation": "Dreamworks animated film style, expressive 3D rendering, stylized proportions, cinematic composition",
    "Disney Classic CGI": "Disney 3D animation style, magical soft glow, pristine rendering, enchanting color palette, polished surfaces",
    # Stylized 3D
    "Low Poly Art": "low poly geometric art, flat shading, limited color palette, minimalist 3D aesthetic, clean angular forms, subtle gradients",
    "Claymation Style": "claymation 3D style, stop-motion aesthetic, subtle fingerprint textures, handcrafted charm, matte clay materials",
    "Anime Cel Shaded": "3D cel shaded anime style, bold outlines, flat color regions, vibrant saturated palette, stylized shading",
    "Paper Craft 3D": "paper craft 3D style, folded origami aesthetic, visible crease lines, layered cutout look, soft shadows, handmade texture",
    # Realistic Rendering
    "Hyperrealistic CGI": "hyperrealistic CGI render, ray traced reflections, photorealistic materials, 8K detail, physically accurate rendering",
    "Unreal Engine Cinematic": "Unreal Engine 5 cinematic render, photorealistic global illumination, nanite geometry, lumen lighting, film grain",
    "Blender Cycles Render": "Blender Cycles photorealistic render, path traced lighting, physically based materials, HDRI environment, volumetric atmosphere",
    "Octane Render Style": "Octane render aesthetic, ultra clean lighting, specular highlights, smooth geometry, commercial product visualization, pristine materials",
    # Game Art Styles
    "Fortnite Art Style": "Fortnite stylized 3D, vibrant cartoon rendering, exaggerated proportions, clean bold colors, playful aesthetic, soft shadows",
    "Overwatch Hero Style": "Overwatch 3D art style, bold stylized rendering, heroic proportions, clean textures, saturated palette",
    "Borderlands Cell Shade": "Borderlands style rendering, heavy black outlines, crosshatch shading, comic book aesthetic, gritty textures, high contrast",
    "Nintendo Figurine": "Nintendo amiibo figure style, glossy plastic materials, clean bright colors, collectible aesthetic, smooth surfaces",
    # Retro 3D
    "PS1 Retro 3D": "PlayStation 1 era 3D graphics, low resolution textures, vertex jitter, affine texture mapping, nostalgic retro CG, limited polygon count",
    "N64 Low Res": "Nintendo 64 style graphics, chunky polygons, blurry textures, fog distance fade, retro game aesthetic, warm CRT colors",
    "Isometric Game Art": "isometric 3D game art, clean orthographic view, crisp pixel-perfect edges, strategy game aesthetic, diorama feel",
    # Aesthetic 3D
    "Vaporwave 3D": "vaporwave 3D aesthetic, neon pink and cyan, chrome reflections, marble busts, grid floors, retro CG graphics, glitch artifacts",
    "Synthwave Neon": "synthwave neon render, glowing wireframes, chrome surfaces, laser grid, 80s sci-fi aesthetic, retrowave style",
    "Cyberpunk Neon City": "cyberpunk 3D aesthetic, holographic advertisements, dystopian atmosphere, high tech grime, futuristic urban decay",
    # Materials & Effects
    "Zbrush Sculpture": "Zbrush digital sculpture render, subsurface clay material, fine surface detail, matcap shading, artistic anatomy",
    "Holographic Display": "holographic 3D display effect, translucent blue glow, scan lines, floating interface aesthetic, sci-fi UI, volumetric projection",
    "Chrome Liquid Metal": "liquid chrome 3D render, mirror reflections, organic flowing metal, caustic light patterns, surreal metallic aesthetic, smooth deformations",
    "Glass Transparency": "glass material 3D render, caustic refractions, prismatic light dispersion, crystal clarity, elegant transparency, studio gradient backdrop",
    "Soft Plastic Toy": "soft vinyl toy render, smooth matte plastic, rounded friendly forms, pastel color palette, designer toy aesthetic, subtle ambient shadows",
    "Inflatable Balloon": "inflatable 3D render, glossy balloon material, soft puffy forms, playful rounded shapes, squeezable aesthetic",
    "Frozen Ice Crystal": "frozen ice 3D render, crystalline subsurface scattering, frost surface detail, prismatic sparkles, frozen aesthetic",
    "Miniature Diorama": "miniature diorama render, tiny scale illusion, tabletop gaming aesthetic, detailed model scenery, handcrafted miniature world",
}

FINE_ART_STYLES = {
    "None": "",
    # Impressionism & Post
    "Impressionist Landscape": "impressionist oil painting, visible brushstrokes, dappled light, soft edges, vibrant complementary colors, en plein air atmosphere",
    "Post-Impressionist": "post-impressionist painting, bold outlined forms, flattened perspective, expressive color patches, structured brushwork, decorative patterning",
    "Pointillist": "pointillist painting, tiny dots of pure color, optical color mixing, shimmering luminosity, scientific color theory, mosaic-like texture",
    # Classical Periods
    "Renaissance Portrait": "renaissance oil painting, sfumato technique, rich earth tones, meticulous glazing, classical composition",
    "Baroque Drama": "baroque oil painting, tenebrism, deep shadows, golden highlights, rich velvety darks, dynamic diagonal composition",
    "Romantic Landscape": "romantic era oil painting, sublime nature, dramatic skies, atmospheric perspective, emotional grandeur, luminous glazes",
    "Dutch Golden Age": "dutch golden age oil painting, luminous realism, rich warm undertones, delicate light rendering, intimate domestic atmosphere, masterful glazing",
    "Neo-Classical": "neoclassical painting, idealized forms, restrained noble colors, balanced symmetrical composition, smooth polished surface",
    "Pre-Raphaelite Detailed": "pre-raphaelite painting, jewel-like colors, intricate botanical details, luminous skin tones, sharp focus throughout, medieval romanticism",
    # Modern Movements
    "Expressionist Bold": "expressionist painting, distorted forms, intense saturated colors, visible aggressive brushwork, emotional color choices, raw psychological intensity",
    "Abstract Expressionist": "abstract expressionist painting, bold gestural brushwork, dripping paint, raw emotional energy, large scale strokes, spontaneous mark-making",
    "Fauvism Vibrant": "fauvist painting, wild non-naturalistic colors, simplified forms, flat bold color areas, spontaneous brushwork, joyful chromatic intensity",
    "Cubist Fragmented": "cubist painting, fragmented geometric forms, multiple simultaneous viewpoints, muted earth tones, flattened picture plane, angular facets",
    "Surrealist Dreamscape": "surrealist oil painting, hyper-detailed rendering, dreamlike juxtapositions, smooth blending, meticulous technique",
    "Pop Art Bold": "pop art painting, flat bright commercial colors, bold black outlines, Ben-Day dots, graphic poster quality, high contrast",
    "Minimalist Color Field": "color field painting, vast flat color areas, subtle color transitions, meditative simplicity, soft edges, immersive chromatic experience",
    # Decorative Styles
    "Art Nouveau Painting": "art nouveau painting, flowing organic lines, decorative botanical motifs, elegant curves, muted jewel tones, ornamental borders",
    "Ukiyo-e Inspired": "ukiyo-e inspired painting, flat color areas, bold outlines, decorative patterns, asymmetrical composition, limited refined color palette",
    # Traditional Media
    "Watercolor Delicate": "delicate watercolor painting, transparent washes, soft wet-on-wet blending, white paper highlights, fluid spontaneous edges, luminous glazes",
    "Gouache Matte": "gouache painting, velvety matte finish, opaque flat colors, crisp edges, subtle tonal gradations, poster-like quality",
    "Acrylic Contemporary": "contemporary acrylic painting, bold saturated colors, sharp edges mixed with soft blending, layered glazes, modern color palette",
    "Palette Knife Textured": "palette knife oil painting, thick impasto texture, bold sculptural strokes, visible paint ridges, dynamic surface, chunky color application",
    # Atmospheric Styles
    "Tonalist Atmospheric": "tonalist painting, muted tonal harmonies, soft atmospheric haze, limited color palette, poetic mood, subtle value gradations",
    "Luminism Serene": "luminist painting, glowing atmospheric light, glassy water reflections, invisible brushwork, serene contemplative mood, precise tonal gradations",
    "Symbolist Mysterious": "symbolist painting, dreamlike atmosphere, muted otherworldly colors, ethereal soft focus, poetic ambiguity",
    # Realism
    "Academic Realism": "academic realist painting, smooth invisible brushwork, careful anatomical rendering, balanced classical composition, refined tonal modeling",
    "Ashcan Realist": "ashcan school painting, gritty urban realism, loose energetic brushwork, dark moody palette, candid street atmosphere, social observation",
    "Contemporary Figurative": "contemporary figurative painting, expressive loose brushwork, unexpected color choices, visible paint texture, modern sensibility, bold mark-making",
}

CINEMATIC_STYLES = {
    "None": "",
    # Film Eras
    "Silent Film Era": "1920s silent film still, black and white, high contrast, film grain, vignette, title card aesthetic",
    "Golden Age Hollywood": "1950s Hollywood film still, technicolor, soft focus glow, rich saturated colors, glamorous aesthetic",
    "70s Gritty Cinema": "1970s film still, grainy 35mm film stock, documentary-style cinematography, lens imperfections, raw realism",
    "80s Blockbuster": "1980s film still, film grain, rich contrast, Panavision look, adventure cinema aesthetic",
    "90s Indie Film": "1990s independent film aesthetic, understated cinematography, authentic raw feel, independent production",
    "2000s Digital Cinema": "early 2000s digital cinema look, crisp detail, clean modern cinematography, digital transition era",
    "Modern Cinematic": "contemporary cinema still, 4K digital clarity, cinematic aspect ratio, professional production, polished aesthetic",
    # Genre Looks
    "Classic Film Noir": "film noir style, high contrast black and white, deep shadows, moody atmospheric, classic crime drama aesthetic",
    "Neo-Noir": "neo-noir aesthetic, high contrast, urban cinematography, modern crime drama, stylized darkness",
    "Spaghetti Western": "spaghetti western film still, widescreen composition, Leone style, frontier aesthetic, dusty gritty feel",
    "Classic Horror": "vintage horror film aesthetic, expressionist shadows, gothic atmosphere, unsettling angles, eerie mood",
    "80s Horror VHS": "1980s horror film still, VHS texture, grimy film grain, practical effects look, tracking line artifacts",
    "Slasher Film": "slasher film aesthetic, dark shadows, suspenseful mood, grainy texture, horror tension",
    "Sci-Fi Neon Future": "1980s sci-fi film still, Blade Runner aesthetic, dystopian future, futuristic noir",
    "Retro Space Opera": "vintage space opera film still, 1970s sci-fi aesthetic, practical model miniature look, optical effects, matte painting style backgrounds",
    "War Film Grit": "war film cinematography, gritty documentary style, combat intensity, visceral realism",
    "Romantic Comedy Glow": "romantic comedy film aesthetic, flattering aesthetic, dreamy bokeh, lighthearted mood",
    "Epic Historical": "historical epic film still, sweeping widescreen composition, rich costume drama, majestic scale",
    "Martial Arts Cinema": "Hong Kong martial arts film aesthetic, action freeze frame style, kung fu energy, dynamic combat",
    # Director-Inspired Styles
    "Wes Anderson Style": "Wes Anderson film aesthetic, symmetrical centered composition, pastel color palette, whimsical staging, flat frontal framing",
    "Kubrick Cold Precision": "Stanley Kubrick aesthetic, symmetrical one-point perspective, clinical precision, unsettling perfection",
    "Spielberg Magic Hour": "Spielberg cinematic style, silhouette compositions, wonder and awe atmosphere, childhood nostalgia",
    "Tarantino Pulp": "Tarantino film aesthetic, bold saturated colors, retro 70s film grain, stylized violence aesthetic, pulp fiction energy",
    "David Lynch Surreal": "David Lynch aesthetic, uncanny dreamlike atmosphere, red curtain noir, unsettling suburban mundane",
    "Ridley Scott Atmospheric": "Ridley Scott cinematography, atmospheric haze, detailed production design, epic moody scale",
    "Villeneuve Modern Epic": "Denis Villeneuve aesthetic, vast minimalist compositions, desaturated muted colors, architectural framing, contemplative stillness",
    "Nolan IMAX Grandeur": "Christopher Nolan IMAX aesthetic, sharp clarity, towering practical scale, immersive scope",
    "Wong Kar-wai Mood": "Wong Kar-wai aesthetic, smeared motion blur, step-printed footage look, melancholic urban atmosphere",
    "Terrence Malick Nature": "Terrence Malick cinematography, spiritual contemplative beauty, poetic imagery, nature reverence",
}

AESTHETIC_STYLES = {
    "None": "",
    # Internet Aesthetics
    "Cottagecore": "cottagecore aesthetic, rustic pastoral mood, wildflowers, handmade textures, gentle countryside atmosphere",
    "Dark Academia": "dark academia aesthetic, vintage scholarly atmosphere, aged paper textures, classical education motifs",
    "Light Academia": "light academia aesthetic, classical elegance, soft focus, airy scholarly atmosphere, refined taste",
    "Vaporwave": "vaporwave aesthetic, retro 80s elements, glitch effects, chrome surfaces, Japanese text elements, nostalgic digital",
    "Dreamcore": "dreamcore aesthetic, surreal hazy atmosphere, liminal space feeling, soft unfocused edges, nostalgic uncanny mood",
    "Goblincore": "goblincore aesthetic, mushrooms, woodland atmosphere, naturalistic textures, found objects, chaotic nature appreciation",
    "Cyberpunk Neon": "cyberpunk aesthetic, urban futuristic, holographic reflections, high tech low life, dystopian",
    "Soft Grunge": "soft grunge aesthetic, film grain texture, melancholic mood, faded vintage feel, 90s alternative",
    "Ethereal Fairycore": "fairycore aesthetic, soft iridescent glow, magical sparkles, dreamy bokeh, enchanted forest atmosphere, whimsical fantasy",
    "Liminal Space": "liminal space aesthetic, unsettling empty atmosphere, uncanny familiarity, transitional spaces, eerie calm, abandoned feeling",
    # Trending Visual Styles
    "Film Photography Look": "analog film photography aesthetic, natural film grain, nostalgic feel, authentic imperfections, vintage aesthetic",
    "Clean Minimal": "clean minimal aesthetic, bright white space, simple color palette, modern calm atmosphere, uncluttered elegance",
    "Retro 70s Warmth": "1970s retro aesthetic, soft focus glow, vintage film quality, nostalgic groovy atmosphere, disco era",
    "Y2K Glossy": "Y2K aesthetic, shiny glossy surfaces, baby pink and silver tones, futuristic optimism, chrome reflections, early digital shimmer",
    "Studio Ghibli Mood": "Studio Ghibli inspired atmosphere, soft watercolor tones, peaceful pastoral mood, whimsical warmth, hand-painted feeling",
}

ANIME_STYLES = {
    "None": "",
    # Era-based styles
    "80s Retro Anime": "1980s anime style, bold linework, limited color palette, dramatic shading, film grain texture, vintage animation aesthetic, classic hand-painted cels",
    "90s Classic Anime": "1990s anime style, cel shading, detailed hand-drawn look, warm color palette, VHS aesthetic, nostalgic animation quality, expressive character designs",
    "Early 2000s Anime": "early 2000s anime style, transitional digital coloring, sharp linework, saturated colors, detailed backgrounds, polished cel-shaded look",
    "Modern Digital Anime": "modern digital anime, clean linework, vibrant colors, detailed eyes, soft gradient shading, high production value, crisp edges",
    "Contemporary Anime": "2020s anime style, refined digital art, subtle color gradients, ultra-clean lines, cinematic composition",
    # Studio-inspired styles
    "Ghibli Inspired": "Studio Ghibli style, soft watercolor backgrounds, gentle color palette, detailed natural environments, whimsical character designs, hand-painted aesthetic",
    "Trigger Action Style": "Trigger studio style, dynamic poses, exaggerated perspectives, bold colors, energetic linework, explosive action sequences, stylized proportions",
    "KyoAni Soft Look": "Kyoto Animation style, pastel colors, detailed eyes with reflections, fluid motion aesthetic, gentle gradients, polished character designs",
    "Madhouse Cinematic": "Madhouse studio style, cinematic framing, detailed realistic proportions, dramatic atmosphere, intense action",
    "Ufotable Visual": "Ufotable style, seamless CGI integration, particle effects aesthetic, vibrant glowing colors, dynamic action poses, ultra-detailed backgrounds",
    "Shaft Artistic": "Shaft studio style, abstract geometric backgrounds, dramatic head tilts, unique framing, high contrast colors, avant-garde visual composition",
    # Genre styles
    "Shonen Action Anime": "shonen anime style, dynamic action poses, bold outlines, intense expressions, speed lines, high energy composition",
    "Shoujo Romance Anime": "shoujo anime style, sparkly eyes, soft pastel colors, floral motifs, delicate linework, dreamy atmosphere, elegant character designs, screen tone patterns",
    "Mecha Epic Anime": "mecha anime style, detailed mechanical designs, metallic shading, dynamic perspectives, technological aesthetic, precise linework",
    "Slice of Life Anime": "slice of life anime style, soft colors, detailed everyday backgrounds, gentle expressions, relaxed atmosphere, realistic proportions",
    "Isekai Fantasy Anime": "isekai anime style, fantasy color palette, magical glow effects, detailed armor and costumes, vibrant world design, adventure aesthetic",
    "Horror Anime": "horror anime style, dark muted colors, heavy shadows, unsettling atmosphere, sharp contrasts, psychological tension aesthetic",
    "Sports Anime": "sports anime style, dynamic motion lines, intense expressions, dramatic perspective, sweat and exertion details, high energy action poses",
    # Art technique styles
    "Chibi Cute": "chibi anime style, super deformed proportions, large head small body, cute simplified features, expressive emotions, rounded shapes, playful aesthetic",
    "Watercolor Anime": "watercolor anime style, soft painted textures, bleeding color edges, gentle gradients, artistic brushstroke quality, ethereal atmosphere",
    "Thick Lineart Anime": "bold outline anime style, thick black linework, flat cel shading, simplified shapes, graphic novel aesthetic, high contrast coloring",
    "Soft Shading Anime": "soft shading anime style, airbrush gradients, minimal linework, smooth color transitions, gentle highlights, dreamy quality",
    "Screentone Classic": "manga-inspired anime style, visible screentone patterns, halftone shading, black and white aesthetic with color accents, retro print look",
    "Painterly Anime": "painterly anime style, visible brushstrokes, oil painting texture, rich color depth, artistic blending, semi-realistic rendering",
    # Aesthetic styles
    "Cyberpunk Anime": "cyberpunk anime style, urban aesthetic, holographic effects, high-tech visual elements, futuristic dystopia",
    "Pastel Soft Anime": "pastel anime style, soft muted colors, minimalist shading, dreamy atmosphere, delicate features, cotton candy color palette",
    "Dark Fantasy Anime": "dark fantasy anime style, gothic aesthetic, deep shadows, rich jewel tones, intricate costume details, mystical atmosphere, dramatic composition",
    "Vintage Shoujo": "vintage shoujo anime style, 1970s aesthetic, starry eyes, flowing hair, rose and flower motifs, romantic soft focus, elegant dramatic poses",
    "Minimalist Anime": "minimalist anime style, clean simple lines, flat colors, reduced detail, modern graphic aesthetic, negative space emphasis, stylized simplicity",
}

CARTOON_STYLES = {
    "None": "",
    # Classic American Era (1930s-1960s)
    "Rubber Hose Classic": "1930s rubber hose animation style, bendy noodle limbs, pie-cut eyes, black and white with grey tones, bouncy exaggerated movements, vintage cartoon charm",
    "Golden Age Disney": "classic Disney animation style, fluid motion lines, expressive characters, warm rich colors, hand-painted watercolor backgrounds, 1940s golden age charm",
    "Looney Tunes": "Warner Bros Looney Tunes style, dynamic action poses, bold primary colors, expressive squash and stretch, painted theatrical backgrounds, zany comic energy",
    "UPA Modern": "1950s UPA animation style, simplified geometric shapes, limited color palettes, stylized angular forms, modernist flat design, sophisticated minimalism",
    "Fleischer Studios": "Fleischer Brothers cartoon style, surreal rubbery characters, jazz age aesthetic, rotoscope-influenced movement, urban 1930s backgrounds, dreamlike whimsy",
    # Television Era (1960s-1980s)
    "Hanna-Barbera Classic": "Hanna-Barbera television animation style, simple character designs, limited animation poses, flat bold colors, repeating background elements, 1960s retro charm",
    "Saturday Morning": "1970s Saturday morning cartoon style, bright cheerful colors, simplified designs, action-ready poses, clean bold outlines, nostalgic adventure aesthetic",
    "80s Action Cartoon": "1980s action cartoon style, heroic muscular proportions, dynamic dramatic poses, metallic sheen effects, explosive energy lines, bold saturated colors",
    # Modern American (1990s-2000s)
    "90s Nickelodeon": "1990s Nickelodeon animation style, exaggerated grotesque expressions, bold irregular outlines, garish clashing colors, irreverent gross-out aesthetic, chaotic energy",
    "Cartoon Network 2000s": "Cartoon Network 2000s style, bold clean outlines, flat vibrant colors, angular geometric character design, energetic dynamic poses, digital clean finish",
    "Powerpuff Style": "Powerpuff Girls animation style, super-deformed chibi proportions, huge sparkling eyes, candy-bright pastel colors, geometric simplified shapes, cute action aesthetic",
    "Dexter's Lab": "Dexter's Laboratory style, retro-futuristic designs, angular sharp shapes, bold contrasting colors, 1960s space age aesthetic, comedic exaggerated expressions",
    "Samurai Jack": "Samurai Jack animation style, dramatic cinematic compositions, bold graphic shapes, limited color palettes, stark shadows, stylized minimalist backgrounds, epic atmosphere",
    # Contemporary (2010s-Present)
    "Adventure Time": "Adventure Time animation style, soft rounded shapes, pastel candy colors, simple cute character designs, whimsical fantasy backgrounds, gentle wobbly lines",
    "Steven Universe": "Steven Universe animation style, soft warm color palettes, rounded friendly shapes, anime-influenced expressions, glowing magical effects, inclusive gentle aesthetic",
    "Gravity Falls": "Gravity Falls animation style, expressive character animation, rich forest color palette, mysterious atmospheric backgrounds, Disney-influenced polish, quirky humor",
    "Modern CalArts": "contemporary CalArts animation style, bean-shaped bodies, noodle limbs, rounded simple features, bright flat colors, expressive squash and stretch, cute appeal",
    "Spider-Verse Style": "Spider-Verse animation style, comic book halftone dots, bold graphic outlines, vibrant clashing colors, multiple frame offset effects, dynamic action lines",
    # Adult Animation
    "Simpsons Style": "The Simpsons animation style, yellow skin tones, bulging round eyes, overbite character design, flat bold colors, sitcom staging, satirical everyday scenes",
    "Adult Swim": "Adult Swim animation style, intentionally crude drawings, limited jerky animation, absurdist surreal aesthetic, late-night weird humor, lo-fi charm",
    "Archer Style": "Archer animation style, sleek mid-century modern aesthetic, sophisticated limited animation, bold graphic shapes, stylish fashion illustration influence",
    # International Styles
    "Franco-Belgian": "Franco-Belgian bande dessinee style, clear ligne claire outlines, detailed European backgrounds, expressive cartoony characters, rich watercolor textures, Tintin aesthetic",
    "British Cartoon": "British animation style, gentle pastoral colors, charming wobbly hand-drawn lines, cozy storybook aesthetic, whimsical understated humor, watercolor textures",
    "Soviet Animation": "Soviet Soyuzmultfilm style, painterly fairy tale aesthetic, rich saturated colors, folk art influences, dreamy atmospheric backgrounds, distinctive Eastern European charm",
    "Modern European": "contemporary European animation style, artistic experimental aesthetic, textured hand-crafted look, sophisticated muted palettes, painterly brushwork, auteur sensibility",
    # Specialty Techniques
    "Cel Shaded 3D Toon": "cel-shaded 3D animation style, bold black outlines, flat color shading, anime-influenced rendering, crisp clean edges, video game aesthetic",
    "Flash Web Animation": "Flash web animation style, vector clean lines, limited tweened motion, flat bright colors, internet era aesthetic, Newgrounds nostalgic charm",
    "Indie Animation": "independent animation style, hand-drawn textured lines, artistic experimental aesthetic, personal expressive brushwork, unique auteur vision, festival quality craft",
    "Motion Graphics Toon": "motion graphics cartoon style, flat geometric shapes, smooth vector aesthetics, trendy pastel gradients, modern minimalist design, corporate friendly polish",
}

# Camera angles, shot types, and lens characteristics
CAMERA_SHOTS = {
    "None": "",
    # Shot Distance
    "Extreme Close-Up": "extreme close-up shot, macro detail, tight framing on small details, intimate perspective",
    "Close-Up": "close-up shot, face or detail filling frame, shallow depth of field, intimate framing",
    "Medium Close-Up": "medium close-up shot, head and shoulders framing, conversational distance",
    "Medium Shot": "medium shot, waist-up framing, balanced composition, standard conversational framing",
    "Medium Wide": "medium wide shot, knee-up framing, showing body language and environment context",
    "Wide Shot": "wide shot, full body visible, environmental context, establishing space",
    "Extreme Wide": "extreme wide shot, vast landscape framing, small subject in large environment, epic scale",
    # Camera Angles
    "Eye Level": "eye level camera angle, neutral perspective, natural viewing height, straightforward composition",
    "Low Angle": "low angle shot, camera looking upward, imposing powerful perspective, dramatic heroic feel",
    "High Angle": "high angle shot, camera looking downward, diminishing vulnerable perspective, overview feeling",
    "Bird's Eye": "bird's eye view, directly overhead perspective, abstract patterns, god's eye view",
    "Worm's Eye": "worm's eye view, extreme low angle from ground level, towering dramatic perspective",
    "Dutch Angle": "dutch angle, tilted camera, diagonal horizon line, unsettling dynamic tension",
    "Over the Shoulder": "over the shoulder shot, partial back of head visible, conversational perspective, depth layering",
    # Lens Types
    "Wide Angle Lens": "wide angle lens, expanded field of view, environmental context, slight barrel distortion",
    "Ultra Wide Lens": "ultra wide angle lens, fisheye-like distortion, dramatic spatial exaggeration, curved horizon",
    "Standard Lens": "standard 50mm lens, natural human eye perspective, minimal distortion, realistic proportions",
    "Telephoto Lens": "telephoto lens, compressed perspective, shallow depth of field, background compression",
    "Macro Lens": "macro lens photography, extreme close-up detail, shallow depth of field, revealing tiny details",
    "Tilt-Shift Lens": "tilt-shift lens effect, selective focus plane, miniature appearance, architectural correction",
    "Anamorphic Lens": "anamorphic lens, horizontal lens flares, oval bokeh, widescreen cinematic quality",
    # Special Techniques
    "Bokeh Background": "shallow depth of field, creamy bokeh background, subject isolation, blurred background circles",
    "Deep Focus": "deep focus, everything sharp front to back, large depth of field, environmental clarity",
    "Rack Focus": "rack focus composition, foreground and background elements, selective sharpness, depth layers",
    "Motion Blur": "motion blur effect, sense of movement and speed, dynamic action, streaking trails",
    "Long Exposure": "long exposure photography, light trails, smooth water, time-lapse effect, ghosting motion",
    "Double Exposure": "double exposure effect, overlapping images, dreamlike transparency, artistic layering",
    # Framing & Composition
    "Centered Composition": "centered symmetrical composition, subject in middle, balanced formal framing",
    "Rule of Thirds": "rule of thirds composition, subject off-center, balanced asymmetrical framing, dynamic placement",
    "Leading Lines": "leading lines composition, converging perspective lines, guiding eye through frame, depth",
    "Framing Within Frame": "frame within frame composition, natural framing elements, doorways windows arches, layered depth",
    "Negative Space": "negative space composition, minimalist framing, isolated subject, breathing room, simplicity",
    "Symmetrical": "symmetrical composition, mirror-like balance, architectural precision, formal elegant framing",
    # POV Shots
    "First Person POV": "first person point of view, seeing through subject's eyes, hands visible, immersive perspective",
    "Drone Aerial": "drone aerial photography, elevated perspective, sweeping landscape view, modern aerial vantage",
    "Security Camera": "security camera angle, high corner mounted view, surveillance aesthetic, voyeuristic",
    "Selfie Angle": "selfie camera angle, arm's length perspective, slightly above eye level, casual intimate",
    "Trunk Shot": "trunk shot, low angle looking up from confined space, framed by surrounding edges",
    # Additional Techniques
    "Fisheye Peephole": "subtle fisheye lens, rounded edges, slight barrel distortion, voyeuristic peephole view",
    "Lens Flare": "lens flare effect, bright light streaks, cinematic light artifacts, sun hitting lens",
    "Establishing Shot": "establishing shot, wide environmental context, scene-setting composition, location reveal",
    "Handheld Camera": "handheld camera feel, subtle motion, documentary authenticity, organic movement",
    "Freeze Frame": "freeze frame moment, action captured mid-motion, dramatic pause, suspended time",
    "Split Diopter": "split diopter shot, dual focus planes, foreground and background both sharp, De Palma style",
}

# =============================================================================
# NEW MODULAR CATEGORIES
# =============================================================================

LIGHTING_STYLES = {
    "None": "",
    # Classic Portrait Lighting
    "Rembrandt Lighting": "Rembrandt lighting, triangular highlight on cheek, dramatic shadows, painterly quality",
    "Butterfly Lighting": "butterfly lighting, shadow under nose, glamorous Hollywood style, even face illumination",
    "Loop Lighting": "loop lighting, small shadow beside nose, flattering portrait light, soft directional",
    "Split Lighting": "split lighting, half face illuminated, dramatic contrast, moody atmosphere",
    "Broad Lighting": "broad lighting, lit side toward camera, wider face appearance, gentle shadows",
    "Short Lighting": "short lighting, shadow side toward camera, slimming effect, dramatic depth",
    # Dramatic & Artistic
    "Rim Light": "rim lighting, glowing edge outline, subject separation, ethereal halo effect",
    "Backlit": "backlit, light source behind subject, silhouette edges, glowing atmosphere",
    "Chiaroscuro": "chiaroscuro lighting, extreme light-dark contrast, dramatic renaissance style, deep shadows",
    "High Key": "high key lighting, bright overall illumination, minimal shadows, clean airy feel",
    "Low Key": "low key lighting, predominantly dark, selective highlights, moody atmospheric",
    "Silhouette": "silhouette lighting, subject in complete shadow, bright background, dramatic shape",
    # Natural Light
    "Soft Diffused": "soft diffused lighting, gentle even illumination, no harsh shadows, flattering",
    "Harsh Direct": "harsh direct lighting, strong defined shadows, high contrast, dramatic edges",
    "Natural Window": "natural window light, soft directional, gentle shadows, intimate atmosphere",
    "Dappled Light": "dappled light through foliage, organic shadow patterns, natural forest feel",
    "Overcast Ambient": "overcast ambient light, soft even illumination, muted shadows, gentle mood",
    # Artificial & Stylized
    "Neon Glow": "neon glow lighting, vibrant colored light, urban night aesthetic, electric atmosphere",
    "Candlelight": "candlelight illumination, warm flickering glow, intimate romantic atmosphere, soft shadows",
    "Studio Three-Point": "studio three-point lighting, key fill and back lights, professional setup, balanced illumination",
    "Ring Light": "ring light illumination, circular catchlights in eyes, even face lighting, beauty aesthetic",
    "Stage Spotlight": "stage spotlight, dramatic pool of light, theatrical atmosphere, dark surroundings",
    # Atmospheric & Environmental
    "Volumetric Light": "volumetric lighting, visible light rays, god rays through atmosphere, dramatic shafts",
    "Moonlight": "moonlight illumination, cool silver-blue tones, night atmosphere, soft lunar glow",
    "Firelight": "firelight illumination, warm orange flickering, campfire atmosphere, dancing shadows",
    "Underwater Caustics": "underwater caustic lighting, rippling light patterns, aquatic atmosphere, dancing reflections",
    # Color Temperature
    "Warm Tungsten": "warm tungsten lighting, orange-yellow tones, cozy indoor atmosphere, incandescent glow",
    "Cool Daylight": "cool daylight lighting, blue-white tones, crisp natural feel, clean illumination",
    "Mixed Color Temperature": "mixed color temperature lighting, warm and cool contrast, cinematic tension, color complexity",
    "Practical Lights": "practical lighting from visible sources, realistic scene illumination, natural motivation",
}

TIME_OF_DAY = {
    "None": "",
    # Dawn & Morning
    "Pre-Dawn": "pre-dawn light, deep blue darkness giving way to first light, quiet anticipation",
    "Dawn": "dawn light, soft pink and orange emerging, gentle awakening atmosphere, delicate colors",
    "Sunrise": "sunrise lighting, warm golden orange sky, long soft shadows, hopeful atmosphere",
    "Golden Hour Morning": "morning golden hour, warm directional sunlight, long shadows, magical glow",
    "Early Morning": "early morning light, crisp clear illumination, fresh atmosphere, soft shadows",
    # Midday
    "Late Morning": "late morning light, bright clear illumination, moderate shadows, energetic feel",
    "High Noon": "high noon sunlight, overhead illumination, short shadows, bright harsh light",
    "Midday Overcast": "midday overcast, soft diffused light, no harsh shadows, even illumination",
    "Early Afternoon": "early afternoon light, warm direct sunlight, defined shadows, active atmosphere",
    # Evening
    "Late Afternoon": "late afternoon light, warming tones, lengthening shadows, relaxed atmosphere",
    "Golden Hour Evening": "evening golden hour, rich warm sunlight, long dramatic shadows, magical warmth",
    "Sunset": "sunset lighting, orange pink purple sky, rim-lit edges, romantic atmosphere",
    "Dusk": "dusk light, fading warm tones, soft ambient glow, transitional atmosphere",
    "Blue Hour": "blue hour light, deep blue ambient, city lights emerging, serene twilight",
    "Twilight": "twilight illumination, purple-blue sky, last traces of daylight, mysterious atmosphere",
    # Night
    "Early Night": "early night, deep blue darkness, artificial lights visible, urban glow",
    "Midnight": "midnight darkness, minimal ambient light, deep shadows, quiet stillness",
    "Late Night": "late night atmosphere, sparse artificial lighting, empty streets, solitary mood",
}

WEATHER_ATMOSPHERE = {
    "None": "",
    # Clear & Sunny
    "Clear Sky": "clear sky, bright direct sunlight, sharp defined shadows, vivid colors",
    "Sunny": "sunny weather, bright cheerful light, warm atmosphere, pleasant day",
    "Partly Cloudy": "partly cloudy sky, dynamic light and shadow, dramatic cloud formations",
    # Overcast & Grey
    "Overcast": "overcast sky, soft diffused light, muted colors, even illumination",
    "Heavy Clouds": "heavy cloud cover, dark moody sky, dramatic atmosphere, stormy potential",
    "Grey Day": "grey day atmosphere, flat even light, subdued colors, contemplative mood",
    # Precipitation
    "Light Rain": "light rain, wet surfaces, gentle droplets, fresh atmosphere",
    "Heavy Rain": "heavy rain, downpour, splashing puddles, dramatic wet atmosphere",
    "Drizzle": "drizzle mist, fine rain particles, soft wet atmosphere, hazy mood",
    "Snow": "snowfall, white flakes descending, winter atmosphere, muffled sounds",
    "Blizzard": "blizzard conditions, heavy snow, low visibility, harsh winter",
    "Sleet": "sleet weather, mixed precipitation, icy wet conditions, harsh atmosphere",
    # Fog & Mist
    "Fog": "fog atmosphere, limited visibility, mysterious depth, soft diffused light",
    "Mist": "mist in air, gentle haze, soft atmospheric perspective, dreamy mood",
    "Morning Fog": "morning fog, lifting mist, ethereal atmosphere, revealing light",
    # Dramatic Weather
    "Thunderstorm": "thunderstorm, dramatic dark clouds, lightning potential, electric atmosphere",
    "Storm Approaching": "approaching storm, dark dramatic sky, tension in atmosphere, wind picking up",
    "Storm Clearing": "clearing storm, dramatic light breaking through, hope emerging, wet surfaces",
    # Atmospheric Particles
    "Haze": "hazy atmosphere, reduced visibility, soft distant details, atmospheric perspective",
    "Dust": "dusty atmosphere, particles in air, warm tones, desert feel",
    "Smoke": "smoky atmosphere, hazy diffusion, dramatic light rays, mysterious mood",
    "Humid": "humid atmosphere, heavy air, slight haze, tropical feel",
    # Special Phenomena
    "Aurora": "aurora in sky, northern lights, ethereal green purple ribbons, magical night",
    "Sandstorm": "sandstorm atmosphere, orange brown haze, limited visibility, harsh desert",
}

COLOR_GRADING = {
    "None": "",
    # Cinematic Looks
    "Teal and Orange": "teal and orange color grading, complementary contrast, cinematic blockbuster look",
    "Bleach Bypass": "bleach bypass look, desaturated with high contrast, gritty film aesthetic",
    "Cross-Processed": "cross-processed color shift, unusual color cast, experimental vintage look",
    "Cinematic LUT": "cinematic color grading, lifted blacks, controlled highlights, film emulation",
    "Blockbuster Color": "blockbuster color grading, saturated vibrant, punchy contrast, commercial appeal",
    # Warm Tones
    "Warm Tones": "warm color palette, orange amber golden hues, cozy inviting atmosphere",
    "Golden Warmth": "golden warm tones, honey amber highlights, nostalgic comfortable feel",
    "Sunset Colors": "sunset color palette, orange pink purple gradient, romantic warm atmosphere",
    "Autumn Palette": "autumn color palette, orange brown gold tones, seasonal warmth",
    # Cool Tones
    "Cool Tones": "cool color palette, blue cyan teal hues, calm serene atmosphere",
    "Ice Blue": "ice blue color grading, cold cyan tones, winter aesthetic, crisp feel",
    "Moonlit Cool": "moonlit cool tones, silver blue highlights, night atmosphere",
    "Ocean Blues": "ocean blue palette, teal aqua navy tones, aquatic atmosphere",
    # Desaturated & Muted
    "Desaturated": "desaturated color grading, muted tones, subdued palette, understated look",
    "Muted Pastels": "muted pastel colors, soft desaturated hues, gentle atmosphere",
    "Faded Film": "faded film colors, lifted blacks, reduced saturation, vintage worn look",
    "Earth Tones": "earth tone palette, brown green tan hues, natural organic feel",
    # Vibrant & Saturated
    "Vibrant Saturated": "vibrant saturated colors, punchy vivid hues, high chromatic intensity",
    "Neon Pop": "neon pop colors, electric vibrant hues, bold saturated palette",
    "Technicolor": "technicolor look, rich saturated primary colors, vintage Hollywood glamour",
    "Candy Colors": "candy color palette, bright sweet pastels, playful cheerful aesthetic",
    # Monochrome & Limited
    "Black and White": "black and white, monochrome tones, classic film aesthetic",
    "Sepia": "sepia toning, warm brown vintage look, nostalgic photograph aesthetic",
    "Duotone": "duotone color grading, two-color palette, graphic stylized look",
    "Split Tone": "split tone grading, different colors in highlights and shadows, artistic look",
    # Stylized
    "High Contrast": "high contrast color grading, deep blacks, bright highlights, punchy",
    "Low Contrast": "low contrast color grading, compressed tonal range, soft gentle look",
    "Vintage Film": "vintage film color grading, aged color shift, nostalgic warmth, soft grain",
    "Modern Clean": "modern clean color grading, neutral balanced tones, professional polish",
    "Moody Dark": "moody dark color grading, rich shadows, selective color, atmospheric",
}

CLIMATE_BIOME = {
    "None": "",
    # Tropical & Warm
    "Tropical": "tropical environment, lush green vegetation, humid warm atmosphere, exotic plants",
    "Rainforest": "rainforest setting, dense jungle canopy, misty humid air, rich biodiversity",
    "Savanna": "savanna landscape, golden grasslands, scattered trees, warm dry atmosphere",
    "Desert": "desert environment, arid sandy terrain, sparse vegetation, harsh sun",
    "Oasis": "oasis setting, water in desert, palm trees, refreshing contrast",
    # Temperate
    "Temperate Forest": "temperate forest, deciduous trees, seasonal foliage, moderate climate",
    "Meadow": "meadow setting, open grassland, wildflowers, gentle peaceful atmosphere",
    "Prairie": "prairie landscape, vast open grassland, rolling hills, big sky",
    "Mediterranean": "Mediterranean climate, warm dry summers, olive trees, coastal beauty",
    "Countryside": "countryside setting, pastoral landscape, rolling green hills, rural charm",
    # Cold & Polar
    "Arctic": "arctic environment, ice and snow, polar cold, stark white landscape",
    "Tundra": "tundra landscape, frozen ground, sparse low vegetation, extreme cold",
    "Taiga": "taiga forest, coniferous trees, cold northern climate, boreal atmosphere",
    "Glacier": "glacier setting, massive ice formations, cold blue tones, ancient ice",
    "Alpine": "alpine environment, high mountain meadows, snow-capped peaks, crisp air",
    # Aquatic & Coastal
    "Coastal": "coastal setting, where land meets sea, beach atmosphere, ocean breeze",
    "Coral Reef": "coral reef environment, underwater tropical, colorful marine life, clear water",
    "Wetland": "wetland environment, marshes and swamps, water-logged terrain, rich wildlife",
    "Mangrove": "mangrove swamp, coastal trees in water, brackish environment, tropical coast",
    # Specialty
    "Mountain": "mountain environment, rocky peaks, high altitude, dramatic elevation",
    "Canyon": "canyon landscape, deep carved gorges, layered rock walls, dramatic depths",
    "Volcanic": "volcanic landscape, lava formations, geothermal features, dramatic terrain",
    "Cave": "cave environment, underground darkness, rock formations, mysterious depths",
    "Urban": "urban environment, city setting, buildings and streets, metropolitan atmosphere",
    "Suburban": "suburban setting, residential neighborhood, houses and yards, community feel",
}

SUBJECT_POSE = {
    "None": "",
    # Standing Poses (pure body positions)
    "Standing Casual": "standing in casual relaxed pose, natural comfortable stance",
    "Standing Confident": "standing in confident pose, strong upright posture, powerful presence",
    "Arms Crossed": "standing with arms crossed, confident or defensive stance",
    "Hands in Pockets": "standing with hands in pockets, relaxed casual demeanor",
    "Hands on Hips": "standing with hands on hips, assertive confident pose",
    "Contrapposto": "contrapposto pose, weight on one leg, classical elegant stance",
    # Ground Poses
    "Sitting Cross-Legged": "sitting cross-legged on ground, relaxed meditative pose",
    "Lying on Back": "lying on back, supine position, relaxed or contemplative",
    "Lying on Side": "lying on side, reclining pose, relaxed comfortable",
    "Prone": "lying face down, prone position, resting or dramatic",
    "Kneeling": "kneeling pose, one or both knees on ground, reverent or ready",
    # Action Poses
    "Walking": "walking pose, mid-stride, natural movement captured",
    "Running": "running pose, dynamic motion, athletic energy",
    "Jumping": "jumping pose, airborne, dynamic energetic movement",
    "Dancing": "dancing pose, expressive movement, graceful motion",
    "Fighting Stance": "fighting stance, martial arts ready position, powerful athletic",
    "Twisting": "twisting pose, body rotation captured, dynamic spiral motion",
    "Falling": "falling pose, gravity-defying moment, dramatic suspension",
    # Portrait Angles
    "Headshot": "headshot framing, face and shoulders, portrait composition",
    "Three-Quarter View": "three-quarter view, angled face, classic portrait angle",
    "Profile": "profile view, side of face, silhouette angle, dramatic",
    "Looking Over Shoulder": "looking over shoulder, twist pose, engaging glance back",
    "Looking Up": "looking upward, aspirational pose, hopeful expression angle",
    "Looking Down": "looking downward, contemplative or shy pose, introspective angle",
    # Expressive Poses
    "Thinking Pose": "thinking pose, hand to chin, contemplative intellectual",
    "Laughing": "laughing pose, joyful expression, genuine mirth captured",
    "Dramatic Reach": "dramatic reaching pose, arm extended, emotional gesture",
    "Embracing": "embracing pose, arms wrapped, intimate protective gesture",
    "Praying": "praying pose, hands together, spiritual contemplative",
    "Meditation": "meditation pose, seated peaceful, serene centered",
    # Professional Poses
    "Power Pose": "power pose, expansive confident stance, authority presence",
    "Business Portrait": "business portrait pose, professional composed stance",
    "Model Pose": "fashion model pose, stylized stance, editorial presence",
    "Candid Natural": "candid natural pose, unposed authentic moment captured",
}

POSE_WITH_PROPS = {
    "None": "",
    # Sitting on Furniture
    "Sitting on Chair": "sitting on chair, formal or casual depending on posture",
    "Sitting on Stool": "sitting on stool, bar or counter height, casual elevated seat",
    "Sitting on Couch": "sitting on couch, relaxed comfortable seating, casual home atmosphere",
    "Sitting on Floor Cushion": "sitting on floor cushion, casual relaxed ground seating",
    # Leaning Poses
    "Leaning Against Wall": "leaning casually against wall, relaxed cool demeanor",
    "Leaning on Railing": "leaning on railing, overlooking view, contemplative stance",
    "Leaning on Table": "leaning on table, casual supportive stance, conversational pose",
    # Lounging & Reclining
    "Lounging on Sofa": "lounging on sofa, relaxed reclined position, casual comfort",
    "Reclining on Bed": "reclining on bed, relaxed intimate setting, comfortable repose",
    "Lying on Grass": "lying on grass, outdoor relaxation, natural casual pose",
    # Perched Poses
    "Perched on Ledge": "perched on ledge, edge sitting, alert casual position",
    "Perched on Windowsill": "perched on windowsill, window seat pose, contemplative",
    # Holding Objects
    "Holding Phone": "holding phone, checking device, modern connected pose",
    "Holding Book": "holding book, reading pose, intellectual engaged",
    "Holding Cup": "holding cup or mug, warm beverage, cozy casual moment",
    # Working Poses
    "Working at Desk": "working at desk, office or study setting, productive pose",
    "Typing on Laptop": "typing on laptop, modern work pose, digital engaged",
    # Vehicle Poses
    "Driving": "driving pose, behind the wheel, hands on steering wheel",
    "Riding Bicycle": "riding bicycle, cycling pose, active transportation",
    "On Motorcycle": "on motorcycle, rider pose, cool adventurous stance",
}

VIBE_ATMOSPHERE = {
    "None": "",
    # Peaceful & Calm
    "Serene": "serene atmosphere, peaceful calm, tranquil stillness",
    "Peaceful": "peaceful atmosphere, gentle calm, undisturbed quiet",
    "Tranquil": "tranquil atmosphere, still water calm, meditative peace",
    # Energetic & Dynamic
    "Energetic": "energetic atmosphere, vibrant high energy, dynamic movement",
    "Dynamic": "dynamic atmosphere, active motion, kinetic energy",
    "Vibrant": "vibrant atmosphere, lively colorful, pulsing with life",
    # Melancholic & Somber
    "Melancholic": "melancholic atmosphere, sad reflective, bittersweet mood",
    "Somber": "somber atmosphere, serious grave, heavy contemplation",
    "Wistful": "wistful atmosphere, yearning nostalgia, gentle longing",
    # Mysterious & Enigmatic
    "Mysterious": "mysterious atmosphere, unknown secrets, intriguing shadows",
    "Enigmatic": "enigmatic atmosphere, puzzling allure, cryptic presence",
    "Secretive": "secretive atmosphere, hidden meanings, concealed depths",
    # Romantic & Intimate
    "Romantic": "romantic atmosphere, love-filled air, passionate mood",
    "Intimate": "intimate atmosphere, close personal, private connection",
    "Tender": "tender atmosphere, gentle caring, soft affection",
    # Playful & Light
    "Playful": "playful atmosphere, fun lighthearted, joyful spirit",
    "Whimsical": "whimsical atmosphere, fanciful quirky, imaginative charm",
    "Lighthearted": "lighthearted atmosphere, carefree easy, cheerful mood",
    # Intense & Dramatic
    "Intense": "intense atmosphere, powerful concentrated, heightened emotion",
    "Dramatic": "dramatic atmosphere, theatrical tension, striking impact",
    "Powerful": "powerful atmosphere, strong forceful, commanding presence",
    # Cozy & Comfortable
    "Cozy": "cozy atmosphere, warm snug, comfortable embrace",
    "Warm": "warm atmosphere, inviting friendly, gentle heat",
    "Comfortable": "comfortable atmosphere, at ease relaxed, settled calm",
    # Eerie & Unsettling
    "Eerie": "eerie atmosphere, creepy strange, unnerving quiet",
    "Unsettling": "unsettling atmosphere, disturbing off-putting, uneasy feeling",
    "Haunting": "haunting atmosphere, ghostly persistent, memorable chill",
    # Nostalgic & Sentimental
    "Nostalgic": "nostalgic atmosphere, memories of past, longing for before",
    "Sentimental": "sentimental atmosphere, emotional memories, heartfelt reflection",
    "Bittersweet": "bittersweet atmosphere, joy mixed with sorrow, complex emotion",
    # Ethereal & Dreamlike
    "Ethereal": "ethereal atmosphere, otherworldly delicate, heavenly light",
    "Dreamlike": "dreamlike atmosphere, surreal hazy, floating unreality",
    "Otherworldly": "otherworldly atmosphere, alien strange, beyond normal",
    # Raw & Authentic
    "Raw": "raw atmosphere, unfiltered genuine, stripped bare emotion",
    "Gritty": "gritty atmosphere, rough textured, harsh reality",
    "Authentic": "authentic atmosphere, real genuine, truthful presence",
    # Luxurious & Elegant
    "Luxurious": "luxurious atmosphere, opulent rich, expensive refined",
    "Opulent": "opulent atmosphere, lavish wealthy, grand abundance",
    "Elegant": "elegant atmosphere, sophisticated graceful, refined beauty",
    # Minimal & Clean
    "Minimalist": "minimalist atmosphere, sparse essential, clean simplicity",
    "Clean": "clean atmosphere, pure uncluttered, fresh orderly",
    "Understated": "understated atmosphere, subtle restrained, quiet sophistication",
    # Chaotic & Overwhelming
    "Chaotic": "chaotic atmosphere, disordered frenzied, wild confusion",
    "Frenzied": "frenzied atmosphere, hectic rushed, manic energy",
    "Overwhelming": "overwhelming atmosphere, intense excessive, sensory overload",
}

# =============================================================================
# LOCATION DICTIONARIES
# =============================================================================

SPORTS_LOCATIONS = {
    "None": "",
    "Stadium": "sports stadium, large arena, crowd seating, athletic venue",
    "Arena": "indoor arena, enclosed sports venue, event space",
    "Gymnasium": "gymnasium, indoor sports facility, basketball court, workout space",
    "Basketball Court": "basketball court, hardwood floor, hoops, court lines",
    "Tennis Court": "tennis court, clay or hard surface, net, baseline",
    "Soccer Field": "soccer field, grass pitch, goal posts, field lines",
    "Baseball Diamond": "baseball diamond, infield dirt, grass outfield, home plate",
    "Ice Rink": "ice rink, frozen surface, hockey or skating venue",
    "Swimming Pool": "swimming pool, lanes, diving boards, aquatic center",
    "Boxing Ring": "boxing ring, ropes, canvas floor, corner posts",
    "Golf Course": "golf course, green fairways, sand bunkers, rolling hills",
    "Race Track": "race track, oval circuit, grandstands, pit lane",
    "Ski Slope": "ski slope, snow-covered mountain, chairlift, alpine terrain",
    "Skate Park": "skate park, ramps, rails, concrete bowls, urban sport",
    "Climbing Wall": "climbing wall, holds, ropes, indoor rock climbing facility",
}

REAL_ESTATE_LOCATIONS = {
    "None": "",
    "Modern Living Room": "modern living room, contemporary furniture, open floor plan, stylish decor",
    "Cozy Bedroom": "cozy bedroom, comfortable bed, warm lighting, intimate space",
    "Luxury Bathroom": "luxury bathroom, marble surfaces, spa-like, elegant fixtures",
    "Gourmet Kitchen": "gourmet kitchen, high-end appliances, island counter, chef's space",
    "Home Office": "home office, desk setup, bookshelves, productive workspace",
    "Dining Room": "dining room, formal table setting, chandelier, entertaining space",
    "Penthouse Apartment": "penthouse apartment, floor-to-ceiling windows, city views, luxury living",
    "Loft Space": "loft space, exposed brick, high ceilings, industrial chic",
    "Studio Apartment": "studio apartment, compact efficient, single room living",
    "Suburban House Exterior": "suburban house exterior, front yard, driveway, residential neighborhood",
    "Mansion Entrance": "mansion entrance, grand foyer, sweeping staircase, opulent entry",
    "Rooftop Terrace": "rooftop terrace, outdoor living, city skyline view, urban oasis",
    "Backyard Patio": "backyard patio, outdoor furniture, garden view, entertaining area",
    "Front Porch": "front porch, rocking chairs, welcoming entrance, neighborhood view",
    "Garage": "garage interior, car space, storage, workshop area",
    "Basement": "basement interior, lower level, recreation or utility space",
    "Walk-in Closet": "walk-in closet, organized storage, clothing displays, dressing area",
    "Sunroom": "sunroom, natural light, plants, glass-enclosed relaxation",
    "Wine Cellar": "wine cellar, bottle racks, stone walls, temperature controlled",
    "Home Theater": "home theater, big screen, plush seating, cinema experience",
}

TRAVEL_LOCATIONS = {
    "None": "",
    "Airport Terminal": "airport terminal, departure gates, travelers, modern architecture",
    "Train Station": "train station, platforms, arrival boards, grand hall",
    "Cruise Ship Deck": "cruise ship deck, ocean view, lounge chairs, vacation atmosphere",
    "Hotel Lobby": "hotel lobby, reception desk, elegant seating, hospitality space",
    "Resort Pool": "resort pool, tropical setting, palm trees, vacation relaxation",
    "Beach Resort": "beach resort, oceanfront, cabanas, tropical paradise",
    "Mountain Lodge": "mountain lodge, rustic interior, fireplace, alpine retreat",
    "Desert Oasis": "desert oasis, palm trees, pool in sand, refuge in arid land",
    "Tropical Island": "tropical island, pristine beach, turquoise water, paradise",
    "European Street": "European street, cobblestones, cafes, historic architecture",
    "Asian Temple": "Asian temple, ornate architecture, incense, spiritual atmosphere",
    "African Safari": "African safari, savanna landscape, wildlife viewing, adventure",
    "Ancient Ruins": "ancient ruins, weathered stone, archaeological site, history",
    "Historic Castle": "historic castle, medieval architecture, towers, fortress walls",
    "Famous Landmark": "famous landmark, iconic structure, tourist destination, recognizable",
    "Boutique Hotel": "boutique hotel, unique design, intimate atmosphere, stylish stay",
    "Hostel Common Room": "hostel common room, travelers, budget accommodation, social space",
    "Campervan Interior": "campervan interior, compact living, road trip, mobile home",
    "Ski Resort": "ski resort, snowy slopes, chalet style, winter vacation",
}

PROFESSION_LOCATIONS = {
    "None": "",
    "Corporate Office": "corporate office, cubicles, fluorescent lighting, business environment",
    "Co-working Space": "co-working space, shared desks, startup vibe, collaborative",
    "Conference Room": "conference room, meeting table, presentation screen, professional",
    "Hospital": "hospital interior, medical equipment, clinical environment, healthcare",
    "Laboratory": "laboratory, scientific equipment, research setting, sterile environment",
    "Workshop": "workshop, tools, workbench, craftsman space, maker environment",
    "Factory Floor": "factory floor, machinery, industrial production, manufacturing",
    "Restaurant Kitchen": "restaurant kitchen, commercial appliances, chef stations, culinary",
    "Retail Store": "retail store, merchandise displays, shopping environment, commercial",
    "Salon": "salon or barbershop, styling chairs, mirrors, grooming space",
    "Courtroom": "courtroom, judge's bench, witness stand, legal setting",
    "Classroom": "classroom, desks, blackboard, educational environment",
    "Library": "library interior, bookshelves, reading areas, quiet study space",
    "Fire Station": "fire station, fire trucks, equipment bays, emergency services",
    "Construction Site": "construction site, scaffolding, building in progress, hard hats",
    "Farm": "farm setting, barn, fields, agricultural environment",
    "Warehouse": "warehouse interior, shelving, inventory, logistics space",
    "Dental Office": "dental office, examination chair, medical equipment, clinical",
}

LEISURE_LOCATIONS = {
    "None": "",
    "Coffee Shop": "coffee shop, cafe interior, barista counter, cozy seating",
    "Restaurant": "restaurant interior, dining tables, ambient lighting, fine dining",
    "Bar": "bar or pub interior, bar counter, stools, drinks, social atmosphere",
    "Movie Theater": "movie theater, rows of seats, big screen, cinema experience",
    "Concert Venue": "concert venue, stage, crowd, live music atmosphere",
    "Nightclub": "nightclub interior, dance floor, DJ booth, neon lights, party",
    "Art Gallery": "art gallery, white walls, framed artwork, exhibition space",
    "Museum": "museum interior, exhibits, displays, cultural institution",
    "Bookstore": "bookstore interior, book displays, reading nooks, literary haven",
    "Park Bench": "park bench, tree-lined path, urban green space, outdoor seating",
    "Garden": "garden setting, flowers, landscaping, natural beauty",
    "Rooftop Bar": "rooftop bar, city views, outdoor seating, evening drinks",
    "Spa": "spa interior, relaxation room, treatment space, wellness atmosphere",
    "Yoga Studio": "yoga studio, mats, mirrors, peaceful exercise space",
    "Arcade": "arcade, video game machines, neon lights, entertainment venue",
}

NATURE_LOCATIONS = {
    "None": "",
    "Forest Clearing": "forest clearing, trees surrounding, dappled sunlight, woodland",
    "Mountain Peak": "mountain peak, summit view, rocky terrain, high altitude",
    "Lakeside": "lakeside, calm water, reflections, peaceful shore",
    "Beach Shore": "beach shore, sand, waves, ocean horizon, coastal",
    "Waterfall": "waterfall, cascading water, mist, natural wonder",
    "Cave Entrance": "cave entrance, rock formation, mysterious opening, natural shelter",
    "Meadow": "meadow, wildflowers, tall grass, open field, pastoral",
    "Desert Dunes": "desert dunes, sand formations, arid landscape, stark beauty",
    "Jungle Path": "jungle path, dense vegetation, tropical forest, adventure trail",
    "Cliffside": "cliffside, rocky edge, dramatic drop, coastal or mountain",
    "Riverbank": "riverbank, flowing water, riverside vegetation, natural waterway",
    "Snowy Field": "snowy field, winter landscape, pristine white, cold beauty",
    "Bamboo Forest": "bamboo forest, tall green stalks, filtered light, Asian nature",
    "Volcanic Landscape": "volcanic landscape, lava rocks, crater, geothermal features",
    "Coral Beach": "coral beach, turquoise water, tropical fish, island paradise",
}

URBAN_LOCATIONS = {
    "None": "",
    "City Street": "city street, buildings, traffic, urban bustle, metropolitan",
    "Alleyway": "alleyway, narrow passage, urban grit, back street",
    "Rooftop": "rooftop, city views, urban elevation, building top",
    "Subway Station": "subway station, platform, underground transit, urban commute",
    "Bus Stop": "bus stop, shelter, urban transit, street corner",
    "Bridge": "bridge, spanning structure, river or road crossing, urban landmark",
    "Plaza": "plaza or square, open urban space, gathering place, city center",
    "Market": "street market, vendors, stalls, urban commerce, local goods",
    "Parking Garage": "parking garage, concrete levels, vehicles, urban structure",
    "Abandoned Building": "abandoned building, decay, urban exploration, forgotten space",
    "Graffiti Wall": "graffiti wall, street art, colorful murals, urban expression",
    "Neon-Lit Street": "neon-lit street, night city, glowing signs, urban nightlife",
    "Pedestrian Crossing": "pedestrian crossing, crosswalk, urban intersection, city movement",
    "Food Truck Row": "food truck row, street food, urban dining, casual outdoor",
    "Chinatown": "Chinatown street, lanterns, Asian signage, cultural district",
}

FANTASY_LOCATIONS = {
    "None": "",
    "Enchanted Forest": "enchanted forest, magical trees, glowing flora, fairy tale woods",
    "Fairy Grove": "fairy grove, tiny beings, mushroom circles, magical glade",
    "Mystical Swamp": "mystical swamp, murky waters, mysterious fog, magical wetland",
    "Dragon's Lair": "dragon's lair, treasure hoard, cave dwelling, mythical beast home",
    "Wizard's Tower": "wizard's tower, magical study, spell books, arcane instruments",
    "Floating Island": "floating island, sky realm, levitating landmass, fantasy aerial",
    "Crystal Cavern": "crystal cavern, gemstone formations, magical glow, underground wonder",
    "Ancient Temple Ruins": "ancient temple ruins, forgotten civilization, magical residue",
    "Magical Library": "magical library, enchanted books, floating tomes, arcane knowledge",
    "Elven Palace": "elven palace, elegant architecture, nature harmony, ethereal beauty",
    "Dwarven Forge": "dwarven forge, underground workshop, mastercraft smithy, mountain home",
    "Haunted Mansion": "haunted mansion, gothic architecture, ghostly presence, spooky estate",
    "Underwater Kingdom": "underwater kingdom, merfolk realm, aquatic palace, ocean depths",
    "Cloud City": "cloud city, sky civilization, floating structures, aerial realm",
    "Dark Dungeon": "dark dungeon, stone corridors, torchlight, underground prison",
    "Mushroom Forest": "mushroom forest, giant fungi, bioluminescence, alien woodland",
    "Sacred Grove": "sacred grove, holy site, ancient trees, spiritual sanctuary",
    "Portal Chamber": "portal chamber, magical gateway, dimensional doorway, arcane transit",
}

SCIFI_LOCATIONS = {
    "None": "",
    "Spaceship Bridge": "spaceship bridge, command center, control panels, view screen",
    "Space Station Corridor": "space station corridor, metal walls, airlocks, zero gravity",
    "Alien Planet Surface": "alien planet surface, strange terrain, foreign sky, extraterrestrial",
    "Cyberpunk Street": "cyberpunk street, neon advertisements, rain-slicked, dystopian urban",
    "Neon-Lit Alley": "neon-lit alley, futuristic back street, glowing signs, cyber noir",
    "Holographic Plaza": "holographic plaza, projected displays, futuristic public space",
    "Laboratory of the Future": "futuristic laboratory, advanced equipment, high-tech research",
    "Android Factory": "android factory, robot assembly, synthetic beings, manufacturing",
    "Virtual Reality Space": "virtual reality space, digital environment, simulated world",
    "Mars Colony": "Mars colony, red planet settlement, dome habitats, space colonization",
    "Lunar Base": "lunar base, moon facility, low gravity, space outpost",
    "Asteroid Mining Facility": "asteroid mining facility, space industry, resource extraction",
    "Dystopian Wasteland": "dystopian wasteland, post-apocalyptic, ruined civilization",
    "Megacity Skyline": "megacity skyline, massive towers, flying vehicles, future metropolis",
    "Underground Bunker": "underground bunker, survival shelter, post-apocalyptic refuge",
    "Hyperspace Tunnel": "hyperspace tunnel, faster-than-light travel, warp space visuals",
    "Alien Marketplace": "alien marketplace, exotic goods, diverse species, interstellar trade",
    "Cryo Chamber": "cryo chamber, suspended animation pods, cold sleep, space travel",
}

HISTORICAL_LOCATIONS = {
    "None": "",
    "Victorian Parlor": "Victorian parlor, ornate furniture, gaslight era, 19th century interior",
    "1920s Speakeasy": "1920s speakeasy, prohibition era, jazz age, hidden bar",
    "Wild West Saloon": "Wild West saloon, swinging doors, wooden interior, frontier bar",
    "Ancient Roman Forum": "ancient Roman forum, columns, marble, classical civilization",
    "Egyptian Temple": "Egyptian temple, hieroglyphics, pharaonic architecture, ancient Nile",
    "Greek Amphitheater": "Greek amphitheater, stone seating, ancient performance venue",
    "Medieval Castle Hall": "medieval castle hall, stone walls, banners, great hall",
    "Renaissance Palazzo": "Renaissance palazzo, Italian villa, frescoes, cultural flowering",
    "Baroque Palace": "Baroque palace, ornate gilding, grand halls, royal opulence",
    "1950s Diner": "1950s diner, chrome and neon, jukeboxes, Americana",
    "1960s Mod Apartment": "1960s mod apartment, pop art, space age furniture, groovy",
    "1970s Disco": "1970s disco, dance floor, mirror ball, funk era nightclub",
    "Samurai Dojo": "samurai dojo, training hall, wooden floors, martial tradition",
    "Viking Longhouse": "Viking longhouse, timber hall, fire pit, Norse dwelling",
    "Aztec Pyramid": "Aztec pyramid, stone temple, Mesoamerican architecture, ancient power",
    "Colonial Mansion": "colonial mansion, historical American, plantation era, antebellum",
    "Art Deco Ballroom": "Art Deco ballroom, geometric design, 1930s glamour, jazz era",
    "Industrial Revolution Factory": "industrial revolution factory, steam age, machinery, Victorian industry",
}

ENTERTAINMENT_LOCATIONS = {
    "None": "",
    "Film Set": "film set, cameras, lights, movie production, behind the scenes",
    "Backstage Theater": "backstage theater, dressing rooms, props, pre-performance",
    "Recording Studio": "recording studio, sound booth, mixing console, music production",
    "TV Broadcast Studio": "TV broadcast studio, cameras, anchor desk, live television",
    "Red Carpet Premiere": "red carpet premiere, velvet rope, photographers, celebrity event",
    "Photo Studio": "photo studio, backdrops, lighting equipment, professional photography",
    "Circus Tent": "circus tent, big top, colorful interior, performance space",
    "Carnival Midway": "carnival midway, game booths, rides, fairground atmosphere",
    "Theme Park Ride": "theme park ride, roller coaster, attraction queue, amusement",
    "Casino Floor": "casino floor, gaming tables, slot machines, gambling venue",
    "Poker Room": "poker room, card tables, chips, high stakes gaming",
    "Comedy Club": "comedy club, small stage, intimate seating, stand-up venue",
    "Fashion Runway": "fashion runway, catwalk, audience seating, haute couture show",
    "Green Room": "green room, performer waiting area, pre-show space",
    "Awards Ceremony Stage": "awards ceremony stage, podium, glamorous presentation",
}

RELIGIOUS_SPIRITUAL_LOCATIONS = {
    "None": "",
    "Gothic Cathedral": "Gothic cathedral, soaring arches, stained glass, sacred architecture",
    "Buddhist Temple": "Buddhist temple, peaceful sanctuary, Buddha statues, meditation",
    "Hindu Shrine": "Hindu shrine, colorful deities, incense, devotional space",
    "Mosque Interior": "mosque interior, prayer hall, geometric patterns, Islamic architecture",
    "Shinto Shrine": "Shinto shrine, torii gate, Japanese spiritual, sacred space",
    "Ancient Stone Circle": "ancient stone circle, standing stones, prehistoric sacred site",
    "Meditation Garden": "meditation garden, zen design, raked sand, contemplative space",
    "Monastery Courtyard": "monastery courtyard, cloisters, peaceful enclosure, religious life",
    "Chapel": "chapel, small church, intimate worship, quiet devotion",
    "Cemetery": "cemetery, gravestones, memorial grounds, final resting place",
    "Mausoleum": "mausoleum, stone tomb, memorial architecture, honored dead",
    "Sacred Waterfall": "sacred waterfall, spiritual site, natural holy place, purification",
}

# Combined location dictionary for category selection
ALL_LOCATIONS = {
    "Sports": SPORTS_LOCATIONS,
    "Real Estate": REAL_ESTATE_LOCATIONS,
    "Travel": TRAVEL_LOCATIONS,
    "Profession": PROFESSION_LOCATIONS,
    "Leisure": LEISURE_LOCATIONS,
    "Nature": NATURE_LOCATIONS,
    "Urban": URBAN_LOCATIONS,
    "Fantasy": FANTASY_LOCATIONS,
    "Sci-Fi": SCIFI_LOCATIONS,
    "Historical": HISTORICAL_LOCATIONS,
    "Entertainment": ENTERTAINMENT_LOCATIONS,
    "Religious/Spiritual": RELIGIOUS_SPIRITUAL_LOCATIONS,
}

# =============================================================================
# CLOTHING/OUTFIT DICTIONARIES
# =============================================================================

OUTFIT_GENERAL = {
    "None": "",
    "Casual Outfit": "casual outfit, relaxed everyday clothing, comfortable style",
    "Sportswear": "sportswear, athletic clothing, activewear, workout outfit",
    "Business Casual": "business casual outfit, smart relaxed, office appropriate",
    "Formal Attire": "formal attire, dressy elegant, special occasion clothing",
    "Evening Wear": "evening wear, glamorous outfit, night event attire",
    "Sleepwear": "sleepwear, pajamas, loungewear, comfortable home clothing",
    "Streetwear": "streetwear, urban fashion, trendy casual, youth style",
    "Athleisure": "athleisure, sporty casual, athletic-inspired everyday wear",
    "Bohemian Style": "bohemian style, boho outfit, free-spirited flowing fashion",
    "Vintage Outfit": "vintage outfit, retro clothing, period fashion, nostalgic style",
    "Minimalist Outfit": "minimalist outfit, simple clean lines, understated fashion",
    "Punk Alternative": "punk alternative outfit, edgy rebellious, counterculture style",
    "Preppy": "preppy outfit, classic collegiate, polished traditional",
    "Smart Casual": "smart casual, elevated relaxed, refined but comfortable",
    "Workwear": "workwear, durable practical, labor-ready clothing",
    "Uniform": "uniform, standardized outfit, professional or institutional dress",
    "Swimwear": "swimwear, bathing suit, beach or pool attire",
    "Winter Wear": "winter wear, cold weather clothing, layered warmth",
    "Summer Outfit": "summer outfit, light breathable, warm weather fashion",
    "Loungewear": "loungewear, comfortable home clothes, relaxed casual",
}

OUTFIT_MASCULINE = {
    "None": "",
    "Suit and Tie": "suit and tie, formal menswear, business professional",
    "Dress Shirt and Slacks": "dress shirt and slacks, smart office wear, professional",
    "Jeans and T-Shirt": "jeans and t-shirt, casual classic, everyday menswear",
    "Polo Shirt and Chinos": "polo shirt and chinos, smart casual, refined relaxed",
    "Hoodie and Joggers": "hoodie and joggers, comfortable casual, athletic leisure",
    "Leather Jacket Outfit": "leather jacket outfit, rugged cool, biker style",
    "Tuxedo": "tuxedo, black tie formal, elegant evening menswear",
    "Casual Blazer Look": "casual blazer look, smart jacket, elevated casual",
    "Tank Top and Shorts": "tank top and shorts, summer casual, relaxed warm weather",
    "Athletic Wear": "athletic wear, gym clothes, sports performance outfit",
    "Denim Jacket Outfit": "denim jacket outfit, casual layered, classic Americana",
    "Sweater and Jeans": "sweater and jeans, cozy casual, autumn comfortable",
    "Hawaiian Shirt": "Hawaiian shirt, tropical print, vacation casual",
    "Flannel Shirt Outfit": "flannel shirt outfit, plaid casual, rugged comfortable",
    "Henley and Chinos": "henley shirt and chinos, relaxed masculine, casual refined",
}

OUTFIT_FEMININE = {
    "None": "",
    "Cocktail Dress": "cocktail dress, semi-formal, elegant party attire",
    "Sundress": "sundress, light summer dress, casual feminine, warm weather",
    "Maxi Dress": "maxi dress, long flowing dress, bohemian elegant",
    "Blouse and Skirt": "blouse and skirt, feminine professional, classic pairing",
    "Jeans and Crop Top": "jeans and crop top, casual trendy, youthful style",
    "Business Suit Feminine": "feminine business suit, professional tailored, power dressing",
    "Casual Cardigan Outfit": "casual cardigan outfit, cozy layered, comfortable chic",
    "Evening Gown": "evening gown, formal long dress, glamorous elegant",
    "Romper": "romper or jumpsuit, one-piece casual, playful feminine",
    "Tennis Skirt Outfit": "tennis skirt outfit, sporty feminine, athletic chic",
    "Sweater Dress": "sweater dress, cozy feminine, autumn elegant",
    "Leather Pants Outfit": "leather pants outfit, edgy sophisticated, bold fashion",
    "Flowy Bohemian": "flowy bohemian outfit, romantic layers, free-spirited style",
    "Athletic Set": "athletic set, matching workout, sporty coordinated",
    "Oversized Sweater Look": "oversized sweater look, cozy relaxed, comfortable chic",
}

# Hierarchical style organization
# Main categories -> Subcategories -> Styles
STYLE_HIERARCHY = {
    "Photorealistic": {
        "Photography": PHOTO_STYLES,
        "Cinematic": CINEMATIC_STYLES,
        "Aesthetic & Mood": AESTHETIC_STYLES,
        "Fine Art": FINE_ART_STYLES,
    },
    "Illustration/Anime/Cartoon": {
        "Illustration": ILLUSTRATION_STYLES,
        "Anime": ANIME_STYLES,
        "Cartoon": CARTOON_STYLES,
    },
    "Digital and 3D": {
        "Digital & 3D": DIGITAL_3D_STYLES,
    },
}

# Flat dictionary for backwards compatibility
ALL_STYLES = {
    "Photography": PHOTO_STYLES,
    "Illustration": ILLUSTRATION_STYLES,
    "Anime": ANIME_STYLES,
    "Cartoon": CARTOON_STYLES,
    "Digital & 3D": DIGITAL_3D_STYLES,
    "Fine Art": FINE_ART_STYLES,
    "Cinematic": CINEMATIC_STYLES,
    "Aesthetic & Mood": AESTHETIC_STYLES,
}

# =============================================================================
# NODE DEFINITION
# =============================================================================

class DonutPromptInjection:
    """
    Injects style prompts into user prompts.
    Can place style before or after the user's prompt.
    Includes camera/shot selection independent of artistic style.
    Chainable for combining multiple styles.

    Hierarchical style selection:
    - Main Category (Photorealistic, Illustration/Anime/Cartoon, Digital and 3D)
    - Subcategory (Photography, Cinematic, Anime, etc.)
    - Style (specific style within subcategory)

    Random options:
    - Random main category: picks random subcategory and style
    - Random subcategory: picks random style from that subcategory
    - Random style: picks that specific random style
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Main categories with Random option
        main_categories = ["Random"] + list(STYLE_HIERARCHY.keys())

        # Get first main category's subcategories for default
        first_main = list(STYLE_HIERARCHY.keys())[0]
        default_subcategories = ["Random"] + list(STYLE_HIERARCHY[first_main].keys())

        # Get first subcategory's styles for default
        first_sub = list(STYLE_HIERARCHY[first_main].keys())[0]
        default_styles = ["Random"] + list(STYLE_HIERARCHY[first_main][first_sub].keys())

        camera_shots = ["Random"] + list(CAMERA_SHOTS.keys())
        lighting_options = ["Random"] + list(LIGHTING_STYLES.keys())
        time_options = ["Random"] + list(TIME_OF_DAY.keys())
        weather_options = ["Random"] + list(WEATHER_ATMOSPHERE.keys())
        color_options = ["Random"] + list(COLOR_GRADING.keys())
        climate_options = ["Random"] + list(CLIMATE_BIOME.keys())
        pose_options = ["Random"] + list(SUBJECT_POSE.keys())
        pose_with_props_options = ["Random"] + list(POSE_WITH_PROPS.keys())
        vibe_options = ["Random"] + list(VIBE_ATMOSPHERE.keys())
        location_categories = list(ALL_LOCATIONS.keys())
        default_locations = ["Random"] + list(ALL_LOCATIONS[location_categories[0]].keys())
        outfit_options = ["Random"] + list(OUTFIT_GENERAL.keys())
        outfit_masc_options = ["Random"] + list(OUTFIT_MASCULINE.keys())
        outfit_fem_options = ["Random"] + list(OUTFIT_FEMININE.keys())

        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "main_category": (main_categories, {"default": main_categories[1]}),
                "subcategory": (default_subcategories, {"default": "None"}),
                "style": (default_styles, {"default": "None"}),
                "camera": (camera_shots, {"default": "None"}),
                "lighting": (lighting_options, {"default": "None"}),
                "time_of_day": (time_options, {"default": "None"}),
                "weather": (weather_options, {"default": "None"}),
                "color_grade": (color_options, {"default": "None"}),
                "climate": (climate_options, {"default": "None"}),
                "pose": (pose_options, {"default": "None"}),
                "pose_with_props": (pose_with_props_options, {"default": "None"}),
                "vibe": (vibe_options, {"default": "None"}),
                "location_category": (location_categories, {"default": location_categories[0]}),
                "location": (default_locations, {"default": "None"}),
                "outfit": (outfit_options, {"default": "None"}),
                "outfit_masculine": (outfit_masc_options, {"default": "None"}),
                "outfit_feminine": (outfit_fem_options, {"default": "None"}),
                "order": (["Style First", "Prompt First"], {"default": "Style First"}),
                "separator": ("STRING", {"default": ", "}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("prompt", "style_preview",)
    FUNCTION = "execute"
    CATEGORY = "donutnodes"

    # This makes the dropdowns update based on hierarchy selection
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # Allow all values during transitions - the execute method handles fallbacks
        # This is necessary because dynamic dropdowns may have stale values during updates
        return True

    def _get_random_or_value(self, value, options_dict, rng):
        """Helper to get a value from a dict, handling Random selection."""
        if value == "Random":
            available = [k for k in options_dict.keys() if k != "None"]
            if available:
                selected = rng.choice(available)
                return selected, options_dict[selected]
            return "None", ""
        elif value in options_dict:
            return value, options_dict[value]
        return value, ""

    def _resolve_style_hierarchy(self, main_category, subcategory, style, rng):
        """
        Resolve the hierarchical style selection, handling Random at each level.

        Returns: (actual_main_cat, actual_subcat, actual_style, style_prompt)
        """
        actual_main = main_category
        actual_sub = subcategory
        actual_style = style

        # Handle Random main category - pick random from all subcategories
        if main_category == "Random":
            actual_main = rng.choice(list(STYLE_HIERARCHY.keys()))

        # Get subcategories for the selected main category
        subcategories = STYLE_HIERARCHY.get(actual_main, {})

        # Handle Random subcategory
        if subcategory == "Random" or subcategory not in subcategories:
            available_subs = list(subcategories.keys())
            if available_subs:
                actual_sub = rng.choice(available_subs)
            else:
                return actual_main, "None", "None", ""

        # Get styles for the selected subcategory
        styles_dict = subcategories.get(actual_sub, {})

        # Handle Random style
        if style == "Random" or style not in styles_dict:
            available_styles = [s for s in styles_dict.keys() if s != "None"]
            if available_styles:
                actual_style = rng.choice(available_styles)
            else:
                return actual_main, actual_sub, "None", ""

        # Get the style prompt
        style_prompt = styles_dict.get(actual_style, "")

        return actual_main, actual_sub, actual_style, style_prompt

    def execute(self, prompt, main_category, subcategory, style, camera, lighting, time_of_day,
                weather, color_grade, climate, pose, pose_with_props, vibe,
                location_category, location, outfit, outfit_masculine, outfit_feminine,
                order, separator, seed):
        # Initialize random with seed for reproducibility
        rng = random.Random(seed)

        # Resolve hierarchical style selection
        actual_main, actual_sub, actual_style, style_prompt = self._resolve_style_hierarchy(
            main_category, subcategory, style, rng
        )

        # Get all other category prompts using helper
        actual_camera, camera_prompt = self._get_random_or_value(camera, CAMERA_SHOTS, rng)
        actual_lighting, lighting_prompt = self._get_random_or_value(lighting, LIGHTING_STYLES, rng)
        actual_time, time_prompt = self._get_random_or_value(time_of_day, TIME_OF_DAY, rng)
        actual_weather, weather_prompt = self._get_random_or_value(weather, WEATHER_ATMOSPHERE, rng)
        actual_color, color_prompt = self._get_random_or_value(color_grade, COLOR_GRADING, rng)
        actual_climate, climate_prompt = self._get_random_or_value(climate, CLIMATE_BIOME, rng)
        actual_pose, pose_prompt = self._get_random_or_value(pose, SUBJECT_POSE, rng)
        actual_pose_props, pose_props_prompt = self._get_random_or_value(pose_with_props, POSE_WITH_PROPS, rng)
        actual_vibe, vibe_prompt = self._get_random_or_value(vibe, VIBE_ATMOSPHERE, rng)

        # Handle location (similar to style handling)
        location_prompt = ""
        actual_location = location
        if location == "Random":
            available_locs = [l for l in ALL_LOCATIONS[location_category].keys() if l != "None"]
            if available_locs:
                actual_location = rng.choice(available_locs)
                location_prompt = ALL_LOCATIONS[location_category][actual_location]
        elif location_category in ALL_LOCATIONS and location in ALL_LOCATIONS[location_category]:
            location_prompt = ALL_LOCATIONS[location_category][location]

        actual_outfit, outfit_prompt = self._get_random_or_value(outfit, OUTFIT_GENERAL, rng)
        actual_outfit_masc, outfit_masc_prompt = self._get_random_or_value(outfit_masculine, OUTFIT_MASCULINE, rng)
        actual_outfit_fem, outfit_fem_prompt = self._get_random_or_value(outfit_feminine, OUTFIT_FEMININE, rng)

        # Build the style preview (what gets injected)
        preview_parts = []
        if style_prompt:
            preview_parts.append(f"[{actual_main} > {actual_sub} > {actual_style}]: {style_prompt}")
        if camera_prompt:
            preview_parts.append(f"[Camera: {actual_camera}]: {camera_prompt}")
        if lighting_prompt:
            preview_parts.append(f"[Lighting: {actual_lighting}]: {lighting_prompt}")
        if time_prompt:
            preview_parts.append(f"[Time: {actual_time}]: {time_prompt}")
        if weather_prompt:
            preview_parts.append(f"[Weather: {actual_weather}]: {weather_prompt}")
        if color_prompt:
            preview_parts.append(f"[Color: {actual_color}]: {color_prompt}")
        if climate_prompt:
            preview_parts.append(f"[Climate: {actual_climate}]: {climate_prompt}")
        if pose_prompt:
            preview_parts.append(f"[Pose: {actual_pose}]: {pose_prompt}")
        if pose_props_prompt:
            preview_parts.append(f"[Pose Props: {actual_pose_props}]: {pose_props_prompt}")
        if vibe_prompt:
            preview_parts.append(f"[Vibe: {actual_vibe}]: {vibe_prompt}")
        if location_prompt:
            preview_parts.append(f"[Location: {actual_location}]: {location_prompt}")
        if outfit_prompt:
            preview_parts.append(f"[Outfit: {actual_outfit}]: {outfit_prompt}")
        if outfit_masc_prompt:
            preview_parts.append(f"[Outfit Masc: {actual_outfit_masc}]: {outfit_masc_prompt}")
        if outfit_fem_prompt:
            preview_parts.append(f"[Outfit Fem: {actual_outfit_fem}]: {outfit_fem_prompt}")
        style_preview = "\n".join(preview_parts) if preview_parts else "(no style selected)"

        # Combine all style components
        injection_parts = [p for p in [
            style_prompt, camera_prompt, lighting_prompt, time_prompt,
            weather_prompt, color_prompt, climate_prompt, pose_prompt,
            pose_props_prompt, vibe_prompt, location_prompt,
            outfit_prompt, outfit_masc_prompt, outfit_fem_prompt
        ] if p]
        injection = separator.join(injection_parts)

        # If no style/camera/etc selected, just return the prompt
        if not injection:
            return (prompt, style_preview)

        # If no user prompt, just return the style
        if not prompt.strip():
            return (injection, style_preview)

        # Combine based on order preference
        if order == "Style First":
            result = f"{injection}{separator}{prompt}"
        else:
            result = f"{prompt}{separator}{injection}"

        return (result, style_preview)


# For dynamic style list updates based on hierarchy
class DonutPromptInjectionHelper:
    """Helper node that provides style hierarchy information."""

    @classmethod
    def INPUT_TYPES(cls):
        main_categories = list(STYLE_HIERARCHY.keys())
        return {
            "required": {
                "main_category": (main_categories, {"default": main_categories[0]}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("subcategories_list",)
    FUNCTION = "execute"
    CATEGORY = "donutnodes/utils"

    def execute(self, main_category):
        if main_category in STYLE_HIERARCHY:
            subcategories = list(STYLE_HIERARCHY[main_category].keys())
            return (", ".join(subcategories),)
        return ("",)


NODE_CLASS_MAPPINGS = {
    "DonutPromptInjection": DonutPromptInjection,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutPromptInjection": "Donut Prompt Injection",
}
