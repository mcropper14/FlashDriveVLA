/****************************************************************************
** Meta object code from reading C++ file 'sidebar.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.12.8)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "sidebar.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'sidebar.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.12.8. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_SidebarSP_t {
    QByteArrayData data[9];
    char stringdata0[87];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_SidebarSP_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_SidebarSP_t qt_meta_stringdata_SidebarSP = {
    {
QT_MOC_LITERAL(0, 0, 9), // "SidebarSP"
QT_MOC_LITERAL(1, 10, 11), // "updateState"
QT_MOC_LITERAL(2, 22, 0), // ""
QT_MOC_LITERAL(3, 23, 9), // "UIStateSP"
QT_MOC_LITERAL(4, 33, 1), // "s"
QT_MOC_LITERAL(5, 35, 12), // "valueChanged"
QT_MOC_LITERAL(6, 48, 15), // "sunnylinkStatus"
QT_MOC_LITERAL(7, 64, 10), // "ItemStatus"
QT_MOC_LITERAL(8, 75, 11) // "sidebarTemp"

    },
    "SidebarSP\0updateState\0\0UIStateSP\0s\0"
    "valueChanged\0sunnylinkStatus\0ItemStatus\0"
    "sidebarTemp"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_SidebarSP[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       2,   22, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    1,   19,    2, 0x0a /* Public */,

 // slots: parameters
    QMetaType::Void, 0x80000000 | 3,    4,

 // properties: name, type, flags
       6, 0x80000000 | 7, 0x0049500b,
       8, QMetaType::QString, 0x00495003,

 // properties: notify_signal_id
    1879048197,
    1879048197,

       0        // eod
};

void SidebarSP::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<SidebarSP *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->updateState((*reinterpret_cast< const UIStateSP(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::RegisterPropertyMetaType) {
        switch (_id) {
        default: *reinterpret_cast<int*>(_a[0]) = -1; break;
        case 0:
            *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< ItemStatus >(); break;
        }
    }

#ifndef QT_NO_PROPERTIES
    else if (_c == QMetaObject::ReadProperty) {
        auto *_t = static_cast<SidebarSP *>(_o);
        Q_UNUSED(_t)
        void *_v = _a[0];
        switch (_id) {
        case 0: *reinterpret_cast< ItemStatus*>(_v) = _t->sunnylink_status; break;
        case 1: *reinterpret_cast< QString*>(_v) = _t->sidebar_temp_str; break;
        default: break;
        }
    } else if (_c == QMetaObject::WriteProperty) {
        auto *_t = static_cast<SidebarSP *>(_o);
        Q_UNUSED(_t)
        void *_v = _a[0];
        switch (_id) {
        case 0:
            if (_t->sunnylink_status != *reinterpret_cast< ItemStatus*>(_v)) {
                _t->sunnylink_status = *reinterpret_cast< ItemStatus*>(_v);
                Q_EMIT _t->valueChanged();
            }
            break;
        case 1:
            if (_t->sidebar_temp_str != *reinterpret_cast< QString*>(_v)) {
                _t->sidebar_temp_str = *reinterpret_cast< QString*>(_v);
                Q_EMIT _t->valueChanged();
            }
            break;
        default: break;
        }
    } else if (_c == QMetaObject::ResetProperty) {
    }
#endif // QT_NO_PROPERTIES
}

QT_INIT_METAOBJECT const QMetaObject SidebarSP::staticMetaObject = { {
    &Sidebar::staticMetaObject,
    qt_meta_stringdata_SidebarSP.data,
    qt_meta_data_SidebarSP,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *SidebarSP::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *SidebarSP::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_SidebarSP.stringdata0))
        return static_cast<void*>(this);
    return Sidebar::qt_metacast(_clname);
}

int SidebarSP::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = Sidebar::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 1)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 1;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 1)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 1;
    }
#ifndef QT_NO_PROPERTIES
    else if (_c == QMetaObject::ReadProperty || _c == QMetaObject::WriteProperty
            || _c == QMetaObject::ResetProperty || _c == QMetaObject::RegisterPropertyMetaType) {
        qt_static_metacall(this, _c, _id, _a);
        _id -= 2;
    } else if (_c == QMetaObject::QueryPropertyDesignable) {
        _id -= 2;
    } else if (_c == QMetaObject::QueryPropertyScriptable) {
        _id -= 2;
    } else if (_c == QMetaObject::QueryPropertyStored) {
        _id -= 2;
    } else if (_c == QMetaObject::QueryPropertyEditable) {
        _id -= 2;
    } else if (_c == QMetaObject::QueryPropertyUser) {
        _id -= 2;
    }
#endif // QT_NO_PROPERTIES
    return _id;
}
// If you get a compile error in this function it can be because either
//     a) You are using a NOTIFY signal that does not exist. Fix it.
//     b) You are using a NOTIFY signal that does exist (in a parent class) but has a non-empty parameter list. This is a moc limitation.
Q_DECL_UNUSED static void checkNotifySignalValidity_SidebarSP(SidebarSP *t) {
    t->valueChanged();
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
