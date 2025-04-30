from util.import_util import script_imports

script_imports(allow_zluda=False)

# --- START: remover import estático da UI para lazy-load ---
# from modules.ui.TrainUI import TrainUI
# ---  END: import estático da UI  ---

global_ui = None

def main():
    # --- START: import da UI só quando for realmente instanciar a janela ---
    from modules.ui.TrainUI import TrainUI
    # ---  END: lazy-load da UI  ---

    global global_ui
    global_ui = TrainUI()
    global_ui.mainloop()

if __name__ == '__main__':
    main()