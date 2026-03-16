def classifier_structure(freq):
    if freq > 0.12:
        return "CAT1_bandes_fines"

    elif freq > 0.04:
        return "CAT2_bandes_moyennes"

    else:
        return "CAT3_bandes_larges"
