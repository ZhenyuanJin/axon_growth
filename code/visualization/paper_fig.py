





from utils_function.paper_figures import PaperFig

model_timedir_list = []
marmoset_V4_timedir = ''

if len(model_timedir_list) != 4 or not marmoset_V4_timedir:
    raise ValueError(
        'Set the four model_timedir_list entries and marmoset_V4_timedir '
        'from the output of code/run_connection_generation.py.'
    )

paper_fig = PaperFig(
    model_timedir_list=model_timedir_list,
    marmoset_V4_timedir=marmoset_V4_timedir,
)
paper_fig.plot_all()
paper_fig.table_1()
