###################################################################
#############################[ Utils ]#############################
###################################################################

# straight rip from fuzzywuzzy and the good people at SeatGeek

bad_chars = str("").join([chr(i) for i in range(128, 256)])  # ascii dammit!
PY3 = sys.version_info[0] == 3
if PY3:
    translation_table = dict((ord(c), None) for c in bad_chars)

def asciionly(s):
    if PY3:
        return s.translate(translation_table)
    else:
        return s.translate(None, bad_chars)

def asciidammit(s):
    if type(s) is str:
        return asciionly(s)
    elif type(s) is unicode:
        return asciionly(s.encode('ascii', 'ignore'))
    else:
        return asciidammit(unicode(s))