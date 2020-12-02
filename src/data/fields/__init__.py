"""
A :class:`~data.fields.field.Field` is some piece of data instance
that ends up as an array in a model.
"""

from data.fields.field import Field
from data.fields.array_field import ArrayField
from data.fields.adjacency_field import AdjacencyField
from data.fields.label_field import LabelField
from data.fields.list_field import ListField
from data.fields.metadata_field import MetadataField
from data.fields.production_rule_field import ProductionRuleField
from data.fields.sequence_field import SequenceField
from data.fields.sequence_label_field import SequenceLabelField
from data.fields.span_field import SpanField
from data.fields.text_field import TextField
