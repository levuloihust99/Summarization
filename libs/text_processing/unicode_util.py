Cc_category = [
    "\u0000", # <Null>                                      (NUL)           \0
    "\u0001", # <Start of Heading>                          (SOH)           \1
    "\u0002", # <Start of Text>                             (STX)           \2
    "\u0003", # <End of Text>                               (ETX)           \3
    "\u0004", # <End of Transmission>                       (EOT)           \4
    "\u0005", # <Enquiry>                                   (ENQ)           \5
    "\u0006", # <Acknowledge>                               (ACK)           \6
    "\u0007", # <Alert>                                     (BEL)           \a, \7
    "\u0008", # <Backspace>                                 (BS)            \b
    "\u0009", # <Character Tabulation>                      (HT, TAB)       \t          whitespace
    "\u000A", # <End of Line>                               (EOL, LF, NL)   \n          whitespace
    "\u000B", # <Line Tabulation>                           (VT)            \v          whitespace
    "\u000C", # <Form Feed>                                 (FF)            \f          whitespace
    "\u000D", # <Carriage Return>                           (CR)            \r          whitespace
    "\u000E", # <Locking-Shift One>                         (SO)
    "\u000F", # <Locking-Shift Zero>                        (SI)
    "\u0010", # <Data Link Escape>                          (DLE)
    "\u0011", # <Device Control One>                        (DC1)
    "\u0012", # <Device Control Two>                        (DC2)
    "\u0013", # <Device Control Three>                      (DC3)
    "\u0014", # <Device Control Four>                       (DC4)
    "\u0015", # <Negative Acknowledge>                      (NAK)
    "\u0016", # <Synchronous Idle>                          (SYN)
    "\u0017", # <End of Transmission Block>                 (ETB)
    "\u0018", # <Cancel>                                    (CAN)
    "\u0019", # <End of Medium>                             (EOM)
    "\u001A", # <Substitute>                                (SUB)
    "\u001B", # <Escape>                                    (ESC)
    "\u001C", # <File Separator>                            (FS)                        whitespace
    "\u001D", # <Group Separator>                           (GS)                        whitespace
    "\u001E", # <Information Separator Two>                 (RS)                        whitespace
    "\u001F", # <Information Separator One>                 (US)                        whitespace
    # Block separation
    "\u007F", # <Delete>                                    (DEL)
    "\u0080", # <Padding Character>                         (PAD)
    "\u0081", # <High Octet Preset>                         (HOP)
    "\u0082", # <Break Permitted Here>                      (BPH)
    "\u0083", # <No Break Here>                             (NBH)
    "\u0084", # <Index>                                     (IND)
    "\u0085", # <Next Line>                                 (NEL)                       whitespace
    "\u0086", # <Start of Selected Area>                    (SSA)
    "\u0087", # <End of Selected Area>                      (ESA)
    "\u0088", # <Character Tabulation Set)                  (HTS)
    "\u0089", # <Character Tabulation with Justification>   (HTJ)
    "\u008A", # <Line Tabulation Set>                       (VTS)
    "\u008B", # <Partial Line Down>                         (PLD)
    "\u008C", # <Partial Line Backward>                     (PLU)
    "\u008D", # <Reverse Index>                             (RI)
    "\u008E", # <Single Shift Two>                          (SS2)
    "\u008F", # <Single Shift Three>                        (SS3)
    "\u0090", # <Device Control String>                     (DCS)
    "\u0091", # <Private Use One>                           (PU1)
    "\u0092", # <Private Use Two>                           (PU2)
    "\u0093", # <Set Transmit State>                        (STS)
    "\u0094", # <Cancel Character>                          (CCH)
    "\u0095", # <Message Waiting>                           (MW)
    "\u0096", # <Start of Guarded Area>                     (SPA)
    "\u0097", # <End of Guarded Area>                       (EPA)
    "\u0098", # <Start of String>                           (SOS)
    "\u0099", # <Single Graphic Character Introducer>       (SGC)
    "\u009A", # <Single Character Introducer>               (SCI)
    "\u009B", # <Control Sequence Introducer>               (CSI)
    "\u009C", # <String Terminator>                         (ST)
    "\u009D", # <Operating System Command>                  (OSC)
    "\u009E", # <Privacy Message>                           (PM)
    "\u009F", # <Application Program Command>               (APC)
]

Cf_category = [
    "\u00AD", # Soft Hyphen                                     (SHY)
    "\u0600", # Arabic Number Sign
    "\u0601", # Arabic Sign Sanah
    "\u0602", # Arabic Footnote Marker
    "\u0603", # Arabic Sign Safha
    "\u0604", # Arabic Sign Samvat
    "\u0605", # Arabic Number Mark Above
    "\u061C", # Arabic Letter Mark                              (ALM)
    "\u06DD", # Arabic End of Ayah
    "\u070F", # Syriac Abbreviation Mark
    "\u08E2", # Arabic Disputed End of Ayah
    "\u180E", # Mongolian Vowel Separator                       (MVS)
    "\u200B", # Zero Width Space                                (ZWSP)
    "\u200C", # Zero Width Non-Joiner                           (ZWNJ)
    "\u200D", # Zero Width Joiner                               (ZWJ)
    "\u200E", # Left-to-Right Mark                              (LRM)
    "\u200F", # Right-to-Left Mark                              (RLM)
    "\u202A", # Left-to-Right Embedding                         (LRE)
    "\u202B", # Right-to-Left Embedding                         (RLE)
    "\u202C", # Pop Directional Formatting                      (PDF)
    "\u202D", # Left-to-Right Override                          (LRO)
    "\u202E", # Right-to-Left Override                          (RLO)
    "\u2060", # Word Joiner                                     (WJ)
    "\u2061", # Function Application
    "\u2062", # Invisible Times
    "\u2063", # Invisible Separator
    "\u2064", # Invisible Plus
    "\u2066", # Left-to-Right Isolate                           (LRI)
    "\u2067", # Right-to-Left Isolate                           (RLI)
    "\u2068", # First Strong Isolate                            (FSI)
    "\u2069", # Pop Directional Isolate                         (PDI)
    "\u206A", # Inhibit Symmetric Swapping
    "\u206B", # Activate Symmetric Swapping
    "\u206C", # Inhibit Arabic Form Shaping
    "\u206D", # Activate Arabic Form Shaping
    "\u206E", # National Digit Shapes
    "\u206F", # Nominal Digit Shapes
    "\uFEFF", # Zero Width No-Break Space                       (BOM, ZWNBSP)
    "\uFFF9", # Interlinear Annotation Anchor
    "\uFFFA", # Interlinear Annotation Separator
    "\uFFFB", # Interlinear Annotation Terminator
    "\U000110BD", # Kaithi Number Sign
    "\U000110CD", # Kaithi Number Sign Above
    "\U00013430", # Egyptian Hieroglyph Vertical Joiner
    "\U00013431", # Egyptian Hieroglyph Horizontal Joiner
    "\U00013432", # Egyptian Hieroglyph Insert at Top Start
    "\U00013433", # Egyptian Hieroglyph Insert at Bottom Start
    "\U00013434", # Egyptian Hieroglyph Insert at Top End
    "\U00013435", # Egyptian Hieroglyph Insert at Bottom End
    "\U00013436", # Egyptian Hieroglyph Overlay Middle
    "\U00013437", # Egyptian Hieroglyph Begin Segment
    "\U00013438", # Egyptian Hieroglyph End Segment
    "\U0001BCA0", # Shorthand Format Letter Overlap
    "\U0001BCA1", # Shorthand Format Continuing Overlap
    "\U0001BCA2", # Shorthand Format Down Step
    "\U0001BCA3", # Shorthand Format Up Step
    "\U0001D173", # Musical Symbol Begin Beam
    "\U0001D174", # Musical Symbol End Beam
    "\U0001D175", # Musical Symbol Begin Tie
    "\U0001D176", # Musical Symbol End Tie
    "\U0001D177", # Musical Symbol Begin Slur
    "\U0001D178", # Musical Symbol End Slur
    "\U0001D179", # Musical Symbol Begin Phrase
    "\U0001D17A", # Musical Symbol End Phrase
    "\U000E0001", # Language Tag
    "\U000E0020", # Tag Space
    "\U000E0021", # Tag Exclamation Mark
    "\U000E0022", # Tag Quotation Mark
    "\U000E0023", # Tag Number Sign
    "\U000E0024", # Tag Dollar Sign
    "\U000E0025", # Tag Percent Sign
    "\U000E0026", # Tag Ampersand
    "\U000E0027", # Tag Apostrophe
    "\U000E0028", # Tag Left Parenthesis
    "\U000E0029", # Tag Right Parenthesis
    "\U000E002A", # Tag Asterisk
    "\U000E002B", # Tag Plus Sign
    "\U000E002C", # Tag Comma
    "\U000E002D", # Tag Hyphen-Minus
    "\U000E002E", # Tag Full Stop
    "\U000E002F", # Tag Solidus
    "\U000E0030", # Tag Digit Zero
    "\U000E0031", # Tag Digit One
    "\U000E0032", # Tag Digit Two
    "\U000E0033", # Tag Digit Three
    "\U000E0034", # Tag Digit Four
    "\U000E0035", # Tag Digit Five
    "\U000E0036", # Tag Digit Six
    "\U000E0037", # Tag Digit Seven
    "\U000E0038", # Tag Digit Eight
    "\U000E0039", # Tag Digit Nine
    "\U000E003A", # Tag Colon
    "\U000E003B", # Tag Semicolon
    "\U000E003C", # Tag Less-Then Sign
    "\U000E003D", # Tag Equals Sign
    "\U000E003E", # Tag Greater-Than Sign
    "\U000E003F", # Tag Question Mark
    "\U000E0040", # Tag Commercial At
    "\U000E0041", # Tag Latin Capital Letter A
    "\U000E0042", # Tag Latin Capital Letter B
    "\U000E0043", # Tag Latin Capital Letter C
    "\U000E0044", # Tag Latin Capital Letter D
    "\U000E0045", # Tag Latin Capital Letter E
    "\U000E0046", # Tag Latin Capital Letter F
    "\U000E0047", # Tag Latin Capital Letter G
    "\U000E0048", # Tag Latin Capital Letter H
    "\U000E0049", # Tag Latin Capital Letter I
    "\U000E004A", # Tag Latin Capital Letter J
    "\U000E004B", # Tag Latin Capital Letter K
    "\U000E004C", # Tag Latin Capital Letter L
    "\U000E004D", # Tag Latin Capital Letter M
    "\U000E004E", # Tag Latin Capital Letter N
    "\U000E004F", # Tag Latin Capital Letter O
    "\U000E0050", # Tag Latin Capital Letter P
    "\U000E0051", # Tag Latin Capital Letter Q
    "\U000E0052", # Tag Latin Capital Letter R
    "\U000E0053", # Tag Latin Captial Letter S
    "\U000E0054", # Tag Latin Capital Letter T
    "\U000E0055", # Tag Latin Capital Letter U
    "\U000E0056", # Tag Latin Capital Letter V
    "\U000E0057", # Tag Latin Capital Letter W
    "\U000E0058", # Tag Latin Capital Letter X
    "\U000E0059", # Tag Latin Capital Letter Y
    "\U000E005A", # Tag Latin Capital Letter Z
    "\U000E005B", # Tag Left Square Bracket
    "\U000E005C", # Tag Reverse Solidus
    "\U000E005D", # Tag Right Square Bracket
    "\U000E005E", # Tag Circumflex Accent
    "\U000E005F", # Tag Low Line
    "\U000E0060", # Tag Grave Accent
    "\U000E0061", # Tag Latin Small Letter A
    "\U000E0062", # Tag Latin Small Letter B
    "\U000E0063", # Tag Latin Small Letter C
    "\U000E0064", # Tag Latin Small Letter D
    "\U000E0065", # Tag Latin Small Letter E
    "\U000E0066", # Tag Latin Small Letter F
    "\U000E0067", # Tag Latin Small Letter G
    "\U000E0068", # Tag Latin Small Letter H
    "\U000E0069", # Tag Latin Small Letter I
    "\U000E006A", # Tag Latin Small Letter J
    "\U000E006B", # Tag Latin Small Letter K
    "\U000E006C", # Tag Latin Small Letter L
    "\U000E006D", # Tag Latin Small Letter M
    "\U000E006E", # Tag Latin Small Letter N
    "\U000E006F", # Tag Latin Small Letter O
    "\U000E0070", # Tag Latin Small Letter P
    "\U000E0071", # Tag Latin Small Letter Q
    "\U000E0072", # Tag Latin Small Letter R
    "\U000E0073", # Tag Latin Small Letter S
    "\U000E0074", # Tag Latin Small Letter T
    "\U000E0075", # Tag Latin Small Letter U
    "\U000E0076", # Tag Latin Small Letter V
    "\U000E0077", # Tag Latin Small Letter W
    "\U000E0078", # Tag Latin Small Letter X
    "\U000E0079", # Tag Latin Small Letter Y
    "\U000E007A", # Tag Latin Small Letter Z
    "\U000E007B", # Tag Left Curly Bracket
    "\U000E007C", # Tag Vertical Line
    "\U000E007D", # Tag Right Curly Bracket
    "\U000E007E", # Tag Tilde
    "\U000E007F", # Cancel Tag
]

Zl_category = [
    "\u2028", # Line Separator
]

Zp_category = [
    "\u2029", # Paragraph Separator
]

Zs_category = [
    "\u0020", # Space                               (SP)
    "\u00A0", # No-Break Space                      (NBSP)
    "\u1680", # Ogham Space Mark
    "\u2000", # En Quad
    "\u2001", # Em Quad
    "\u2002", # En Space
    "\u2003", # Em Space
    "\u2004", # Three-Per-Em Space
    "\u2005", # Four-Per-Em Space
    "\u2006", # Six-Per-Em Space
    "\u2007", # Figure Space
    "\u2008", # Punctuation Space
    "\u2009", # Thin Space
    "\u200A", # Hair Space
    "\u202F", # Narrow No-Break Space               (NNBSP)
    "\u205F", # Medium Mathematical Space           (MMSP)
    "\u3000", # Ideographic Space
]