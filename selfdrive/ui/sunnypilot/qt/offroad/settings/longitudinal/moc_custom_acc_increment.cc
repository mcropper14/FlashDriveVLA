/****************************************************************************
** Meta object code from reading C++ file 'custom_acc_increment.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.12.8)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "custom_acc_increment.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'custom_acc_increment.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.12.8. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_CustomAccIncrement_t {
    QByteArrayData data[1];
    char stringdata0[19];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_CustomAccIncrement_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_CustomAccIncrement_t qt_meta_stringdata_CustomAccIncrement = {
    {
QT_MOC_LITERAL(0, 0, 18) // "CustomAccIncrement"

    },
    "CustomAccIncrement"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_CustomAccIncrement[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

void CustomAccIncrement::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    Q_UNUSED(_o);
    Q_UNUSED(_id);
    Q_UNUSED(_c);
    Q_UNUSED(_a);
}

QT_INIT_METAOBJECT const QMetaObject CustomAccIncrement::staticMetaObject = { {
    &ExpandableToggleRow::staticMetaObject,
    qt_meta_stringdata_CustomAccIncrement.data,
    qt_meta_data_CustomAccIncrement,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *CustomAccIncrement::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *CustomAccIncrement::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_CustomAccIncrement.stringdata0))
        return static_cast<void*>(this);
    return ExpandableToggleRow::qt_metacast(_clname);
}

int CustomAccIncrement::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = ExpandableToggleRow::qt_metacall(_c, _id, _a);
    return _id;
}
struct qt_meta_stringdata_AccIncrementOptionControl_t {
    QByteArrayData data[1];
    char stringdata0[26];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_AccIncrementOptionControl_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_AccIncrementOptionControl_t qt_meta_stringdata_AccIncrementOptionControl = {
    {
QT_MOC_LITERAL(0, 0, 25) // "AccIncrementOptionControl"

    },
    "AccIncrementOptionControl"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_AccIncrementOptionControl[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

void AccIncrementOptionControl::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    Q_UNUSED(_o);
    Q_UNUSED(_id);
    Q_UNUSED(_c);
    Q_UNUSED(_a);
}

QT_INIT_METAOBJECT const QMetaObject AccIncrementOptionControl::staticMetaObject = { {
    &OptionControlSP::staticMetaObject,
    qt_meta_stringdata_AccIncrementOptionControl.data,
    qt_meta_data_AccIncrementOptionControl,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *AccIncrementOptionControl::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *AccIncrementOptionControl::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_AccIncrementOptionControl.stringdata0))
        return static_cast<void*>(this);
    return OptionControlSP::qt_metacast(_clname);
}

int AccIncrementOptionControl::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = OptionControlSP::qt_metacall(_c, _id, _a);
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
