from mido import Message, MetaMessage, MidiFile, MidiTrack

class MidoMessage():

	@staticmethod
	def note_on( chan = 0, note = 64, vel = 100, dt = 0 ):
		MidoMessage.valid_channel( chan )
		MidoMessage.valid_note( note )
		MidoMessage.valid_velocity( vel )
		MidoMessage.valid_time( dt )
		return Message( "note_on", channel = chan, note = note, velocity = vel, time = dt )

	@staticmethod
	def note_off( chan = 0, note = 64, vel = 100, dt = 0 ):
		MidoMessage.valid_channel( chan )
		MidoMessage.valid_note( note )
		MidoMessage.valid_velocity( vel )
		MidoMessage.valid_time( dt )
		return Message( "note_off", channel = chan, note = note, velocity = vel, time = dt )

	# input indexed 1
	@staticmethod
	def program_change( chan = 0, prog = 1, dt = 0 ):
		MidoMessage.valid_channel( chan )
		MidoMessage.valid_program( prog )
		return Message( "program_change", channel = chan, program = prog, time = dt )

	# input indexed 1
	@staticmethod
	def control_change( chan = 0, control = 0, value = 0, dt = 0 ):
		MidoMessage.valid_channel( chan )
		return Message( "control_change", channel = chan, control = control, value = value, time = dt )

	@staticmethod
	def track_name( name = "default_track" ):
		return MetaMessage( 'track_name', name = name )

	@staticmethod
	def key_signature( key = "C" ):
		MidoMessage.valid_key_signature( key )
		return MetaMessage( 'key_signature', key = key )

	@staticmethod
	def set_tempo( tempo = 500000 ):
		return MetaMessage( 'set_tempo', tempo = tempo )

	@staticmethod
	def time_signature( numerator = 4, denominator = 4, clocks_per_click = 24, notated_32nd_notes_per_beat = 8 ):
		return MetaMessage('time_signature', numerator = numerator, denominator = denominator, 
			                                 clocks_per_click = clocks_per_click,
			                                 notated_32nd_notes_per_beat = notated_32nd_notes_per_beat )

	@staticmethod
	def valid_channel( x ):
		assert( type( x ).__name__ == "int", "Invalid type of channel, expect int" )
		assert( 0 <= x <= 15, "Channel value should between 0 and 15" )

	@staticmethod
	def valid_note( x ):
		assert( type( x ).__name__ == "int", "Invalid type of note, expect int" )
		assert( 0 <= x <= 127, "Note value should between 0 and 127" )

	@staticmethod
	def valid_program( x ):
		assert( type( x ).__name__ == "int", "Invalid type of program, expect int" )
		assert( 1 <= x <= 128, "Channel value should between 1 and 128" )

	@staticmethod
	def valid_velocity( x ):
		assert( type( x ).__name__ == "int", "Invalid type of velocity, expect int" )
		assert( 0 <= x <= 127, "Channel value should between 0 and 127" )
	@staticmethod
	def valid_time( x ):
		assert( type( x ).__name__ == "int", "Invalid type of time, expect int" )

	@staticmethod
	def valid_key_signature( key ):
		valid = [ 'A', 'A#m', 'Ab', 'Abm', 'Am', 'B', 'Bb', 'Bbm', 'Bm', 'C',
				  'C#', 'C#m', 'Cb', 'Cm', 'D', 'D#m', 'Db', 'Dm', 'E', 'Eb', 
				  'Ebm', 'Em', 'F', 'F#', 'F#m', 'Fm', 'G', 'G#m', 'Gb', 'Gm' ]
		assert( key in valid, "Key signature is not in the valid list")

	@staticmethod
	def valid_tempo( tempo ):
		assert( type( tempo ).__name__ == "int", "Invalid type of tempo, expect int" )
		assert( 0 <= tempo <= 16777215, "Tempo value should between 0 and 127" )



def get_instrument_note_range( instrument ):

	return ( 21, 107 )