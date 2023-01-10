class A():
    def __init__(self) -> None:
        self.a = 1
        self.b = 2

    def run(self):
        check_attribute(self)


def check_attribute(A):
    if hasattr(A,'c'):
        print('has attribute')
    else:
        print('no such attribute')


classa = A()
classa.run()

# if hasattr(classa,'c'):
#     print('has attribute')
# else:
#     print('no such attribute')