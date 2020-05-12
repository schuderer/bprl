# Currently only contains a few manual tests of the simulation code
from importlib import reload

import examples.pension as p
reload(p)

s = p.PensionSim()

s.reset()
# s.run()

c = p.PensionInsuranceCompany(s)
c.perform_increment()
print(c)

i = p.Individual(s)
s.entities.append(i)
for _ in range(10):
    #i.resources["cash"].take(10000)
    if i.age > 26:
        i.living_expenses = 50000
    s.run_increment()
po = s.find_entities(p.PublicOpinion)[0]
po.perform_increment()
