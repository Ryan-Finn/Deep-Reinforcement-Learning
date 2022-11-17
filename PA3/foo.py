def main(sll) -> bool:
    if len(sll) == 1:
        return True

    l = [sll[0], sll[1]]
    pal = l[0] == l[1]

    i, j = -1, -2
    for v in sll[2:]:
        pal = l[i] == v or l[j] == v
        l.append(v)

        i -= 1
        j -= 1

        if not pal:
            i = -1
            j = -2

    return pal

    # for i, v in enumerate(l):
    #     if v != l[-1 - i]:
    #         return False
    #     elif i == len(l) // 2:
    #         return True


if __name__ == '__main__':
    print(main([1, 2, 2, 1]) is True)
    print(main([1, 2]) is False)
