from bokeh.io import show, export_png, reset_output
from bokeh.layouts import row, column, gridplot, grid
import Main

#copied in parts from https://dmnfarrell.github.io/bioinformatics/abm-mesa-python

steps = 150
pop = 500
mid_with_hole = [(25,i)for i in range(50) if i not in range(23,28)]
mid_with_hole_horiz = [(i,25)for i in range(50) if i not in range(23,28)]

for i in range(3):
    model = Main.InfectionModel(pop, 50, 50, ptrans=0.25, death_rate=0.01, infected_start=0.01,
                            surrounded=False, walls=[], bed_capacity=0.1, death_untreated=1,
                            tries_to_move=1, chance_to_stay=0, stay_if_infected=0)
    p1 = Main.plot_states_bokeh(model, title='step=%s' % -1, wideness=steps)
    p2 = Main.plot_cells_bokeh(model)
    #export_png(grid([[p1, p2]]), filename="Row/ro{}.png".format(-1))
    for i in range(steps):
        print(i)
        model.step()
        p1 = Main.plot_states_bokeh(model, title='step=%s' % i, wideness=steps)
        p2 = Main.plot_cells_bokeh(model)
        # export_png(grid([[p1,p2]]), filename="Row/row{}.png".format(i))
        reset_output()

    show(grid([[p1,p2]]))






