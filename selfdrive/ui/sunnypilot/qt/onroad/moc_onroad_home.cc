/****************************************************************************
** Meta object code from reading C++ file 'onroad_home.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.12.8)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "onroad_home.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'onroad_home.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.12.8. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_OnroadWindowSP_t {
    QByteArrayData data[7];
    char stringdata0[66];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_OnroadWindowSP_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_OnroadWindowSP_t qt_meta_stringdata_OnroadWindowSP = {
    {
QT_MOC_LITERAL(0, 0, 14), // "OnroadWindowSP"
QT_MOC_LITERAL(1, 15, 17), // "offroadTransition"
QT_MOC_LITERAL(2, 33, 0), // ""
QT_MOC_LITERAL(3, 34, 7), // "offroad"
QT_MOC_LITERAL(4, 42, 11), // "updateState"
QT_MOC_LITERAL(5, 54, 9), // "UIStateSP"
QT_MOC_LITERAL(6, 64, 1) // "s"

    },
    "OnroadWindowSP\0offroadTransition\0\0"
    "offroad\0updateState\0UIStateSP\0s"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_OnroadWindowSP[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       2,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    1,   24,    2, 0x09 /* Protected */,
       4,    1,   27,    2, 0x09 /* Protected */,

 // slots: parameters
    QMetaType::Void, QMetaType::Bool,    3,
    QMetaType::Void, 0x80000000 | 5,    6,

       0        // eod
};

void OnroadWindowSP::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<OnroadWindowSP *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->offroadTransition((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 1: _t->updateState((*reinterpret_cast< const UIStateSP(*)>(_a[1]))); break;
        default: ;
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject OnroadWindowSP::staticMetaObject = { {
    &OnroadWindow::staticMetaObject,
    qt_meta_stringdata_OnroadWindowSP.data,
    qt_meta_data_OnroadWindowSP,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *OnroadWindowSP::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *OnroadWindowSP::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_OnroadWindowSP.stringdata0))
        return static_cast<void*>(this);
    return OnroadWindow::qt_metacast(_clname);
}

int OnroadWindowSP::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = OnroadWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 2)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 2;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 2)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 2;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
