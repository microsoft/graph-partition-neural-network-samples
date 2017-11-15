from __future__ import (absolute_import, division, print_function)

import unittest


class Factory(object):
  _registry = {}

  @classmethod
  def factory(cls, class_name):
    try:
      return cls._registry[class_name]
    except:
      raise ValueError(
          "Unknown class {} from factory {}".format(class_name, cls.__name__))

  @classmethod
  def register(cls, class_name):

    def decorator(sub_class):
      cls._registry[class_name] = sub_class
      return sub_class

    return decorator

  @classmethod
  def get_registry(cls):
    return cls._registry


class ReaderFactory(Factory):
  _registry = {}


class RunnerFactory(Factory):
  _registry = {}


class ModelFactory(Factory):
  _registry = {}


# Unit test
class TestFactory(unittest.TestCase):

  def test_factory(self):

    @ReaderFactory.register("MyReader")
    class MyReader(object):

      def __init__(self, x):
        self.val = x

      def get_val(self):
        return self.val

    my_reader = ReaderFactory.factory("MyReader")(True)
    self.assertTrue(my_reader.get_val())

    with self.assertRaises(ValueError):
      Factory.factory("MyReader")


if __name__ == "__main__":
  unittest.main()
