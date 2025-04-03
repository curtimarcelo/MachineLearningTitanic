def prever_sobrevivencia(pclass, sex, age, family_size, fare, embarked, model):
    passageiro = [[pclass, sex, age, family_size, fare, embarked]]

    predicao = model.predict(passageiro)


    if(predicao == 1):
        return "O passageiro sobreviveu"
    else:
        return "O passageiro não sobreviveu"
