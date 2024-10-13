"""a = [[["a", "b", "c"], ["d", "e", "f"]], [["g", "h", "i"], ["j", "k", "l"]]]
b = [[["o", "p", "q"], ["r", "s", "t"]], [["u", "v", "w"], ["x", "y", "z"]]]


def add(a, b):
    res = []
    for i in range(len(a)):
        res.append(a[i] + b[i])
        print(a[i] + b[i])
    return res


print(list(map(add, a, b)))
print("="*50)
print([add(u[i], v[i]) for u, v in zip(a, b) for i in range(len(u))])
"""

import numpy as np

b = np.array([[1, 2, 3], [4, 5, 6]])

a = [[b, b*2, b*3], [b*4, b*5, b*6]]

print(np.sum(a, axis=0))


def printn(*args):
    for i in args:
        print(i)
        print(list(i))
        print(type(i))
        print("\n")


print("\n===============")


def sort_pyr(pyrs):
    """Sort pyramid by floor"""
    # Initialiser une liste vide pour stocker les groupes
    sorted_pyr = [[] for _ in range(len(pyrs[0]))]

    # Parcourir chaque sous-liste et regrouper les éléments par index
    for floor in pyrs:
        for i, img in enumerate(floor):
            sorted_pyr[i].append(img)

    # Affichage du résultat
    for i, group in enumerate(sorted_pyr):
        print(f"Group {i+1}:")
        for image in group:
            print(image)
    return sorted_pyr


sort_pyr(a)


# Exemple d'une liste de listes de np.array de dimension 2 en uint8
liste = [[np.array([[0, 255], [128, 64]], dtype=np.uint8), np.array([[50, 150], [200, 100]], dtype=np.uint8)],
         [np.array([[75, 25], [175, 125]], dtype=np.uint8), np.array([[10, 90], [30, 240]], dtype=np.uint8)]]

# Conversion de chaque np.array en float
liste_float = [[arr.astype(float) for arr in sous_liste]
               for sous_liste in liste]

# Vérification
for sous_liste in liste_float:
    for arr in sous_liste:
        print(arr)
        print(arr.dtype)  # Doit afficher 'float64' par défaut
