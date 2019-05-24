from scipy.io.wavfile import read as wavread

file_name = "quantized/piano/single/mono_sound/piano_mono_2.wav"
point = 44248
length = 100


rate, data = wavread( file_name )

print( "rate", rate )
print( "data", data.shape )
print( "expected", rate * 4 )

left = data[:,1].tolist()

episode = left[ point:point + length ]

def find_subpattern( lst, pattern ):
	indices = []
	for i in range( len( lst ) - length ):
		if lst[ i: i+length ] == pattern:
			indices.append(i)
	return indices


indexs = find_subpattern( left, episode )
start = indexs[ 0 ]
for i in range( len( indexs ) - 1 ):
	indexs[ i + 1 ]  = indexs[ i + 1 ] - start - 44100*i
print( "indexs", indexs ) 
