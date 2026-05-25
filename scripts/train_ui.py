from util.import_util import script_imports

script_imports()

# remover import estático da UI para lazy-load ---
# from modules.ui.TrainUI import TrainUI

global_ui = None


def main():
    # import da UI só quando for realmente instanciar a janela
    from modules.ui.TrainUI import TrainUI

    global global_ui
    global_ui = TrainUI()
    global_ui.mainloop()


if __name__ == "__main__":
    main()
