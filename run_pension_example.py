# Currently only contains a few manual tests of the simulation code
import sys

from importlib import reload

import examples.pension as p


def notest_pensionsim():
    reload(p)
    s = p.PensionSim()

    s.reset()
    s.run()


def notest_pensioninsurancecompany():
    reload(p)
    s = p.PensionSim()
    s.reset()
    c = p.PensionInsuranceCompany(s)
    c.perform_increment()
    print(c)


def test_individual():
    reload(p)
    s = p.PensionSim()
    s.reset()
    i = p.Individual(s)
    s.entities.append(i)
    for _ in range(10):
        # i.resources["cash"].take(10000)
        if i.age > 26:
            i.living_expenses = 50000
        s.run_increment()
    po = s.find_entities(p.PublicOpinion)[0]
    po.perform_increment()


if __name__ == "__main__":
    func_name: str
    for func_name in dir(sys.modules[__name__]):
        func = getattr(sys.modules[__name__], func_name)
        if callable(func) and func_name.startswith("test_"):
            print(f"Calling {func_name}:")
            func()
            print("###############################")
