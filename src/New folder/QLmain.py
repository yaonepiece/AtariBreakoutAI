import QL, my_game_env as game

game=game.Atarigame(speed=8)
bot=QL.DQN(game.n_features,game.actions,max_memsize=2000,learning_rate=0.01,trainstep=3,refresh_cycle=100)
steps=0
while True:
	bot.play(game)