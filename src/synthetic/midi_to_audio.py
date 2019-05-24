from midi2audio import FluidSynth
import os
# store sound fonts in this dir ~/.fluidsynth/default_sound_font.sf2
sound_font = "gs.sf2"

midi_folder = "midi_piano"

folder = "piano_sound"
os.mkdir( folder )

fs = FluidSynth( sound_font, sample_rate = 44100 )
for i in range( 100 ):
	fs.midi_to_audio( midi_folder + '/' + 'piano_single_' + str( i ) + '.mid', folder + "/" + "output_" + str(i) + ".aiff" )


print("done")