/****************************************************************************
** Meta object code from reading C++ file 'ui.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.12.8)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "ui.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'ui.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.12.8. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_UIStateSP_t {
    QByteArrayData data[13];
    char stringdata0[169];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_UIStateSP_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_UIStateSP_t qt_meta_stringdata_UIStateSP = {
    {
QT_MOC_LITERAL(0, 0, 9), // "UIStateSP"
QT_MOC_LITERAL(1, 10, 20), // "sunnylinkRoleChanged"
QT_MOC_LITERAL(2, 31, 0), // ""
QT_MOC_LITERAL(3, 32, 10), // "subscriber"
QT_MOC_LITERAL(4, 43, 21), // "sunnylinkRolesChanged"
QT_MOC_LITERAL(5, 65, 22), // "std::vector<RoleModel>"
QT_MOC_LITERAL(6, 88, 5), // "roles"
QT_MOC_LITERAL(7, 94, 27), // "sunnylinkDeviceUsersChanged"
QT_MOC_LITERAL(8, 122, 22), // "std::vector<UserModel>"
QT_MOC_LITERAL(9, 145, 5), // "users"
QT_MOC_LITERAL(10, 151, 8), // "uiUpdate"
QT_MOC_LITERAL(11, 160, 1), // "s"
QT_MOC_LITERAL(12, 162, 6) // "update"

    },
    "UIStateSP\0sunnylinkRoleChanged\0\0"
    "subscriber\0sunnylinkRolesChanged\0"
    "std::vector<RoleModel>\0roles\0"
    "sunnylinkDeviceUsersChanged\0"
    "std::vector<UserModel>\0users\0uiUpdate\0"
    "s\0update"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_UIStateSP[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       5,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       4,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   39,    2, 0x06 /* Public */,
       4,    1,   42,    2, 0x06 /* Public */,
       7,    1,   45,    2, 0x06 /* Public */,
      10,    1,   48,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
      12,    0,   51,    2, 0x08 /* Private */,

 // signals: parameters
    QMetaType::Void, QMetaType::Bool,    3,
    QMetaType::Void, 0x80000000 | 5,    6,
    QMetaType::Void, 0x80000000 | 8,    9,
    QMetaType::Void, 0x80000000 | 0,   11,

 // slots: parameters
    QMetaType::Void,

       0        // eod
};

void UIStateSP::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<UIStateSP *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->sunnylinkRoleChanged((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 1: _t->sunnylinkRolesChanged((*reinterpret_cast< std::vector<RoleModel>(*)>(_a[1]))); break;
        case 2: _t->sunnylinkDeviceUsersChanged((*reinterpret_cast< std::vector<UserModel>(*)>(_a[1]))); break;
        case 3: _t->uiUpdate((*reinterpret_cast< const UIStateSP(*)>(_a[1]))); break;
        case 4: _t->update(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (UIStateSP::*)(bool );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&UIStateSP::sunnylinkRoleChanged)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (UIStateSP::*)(std::vector<RoleModel> );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&UIStateSP::sunnylinkRolesChanged)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (UIStateSP::*)(std::vector<UserModel> );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&UIStateSP::sunnylinkDeviceUsersChanged)) {
                *result = 2;
                return;
            }
        }
        {
            using _t = void (UIStateSP::*)(const UIStateSP & );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&UIStateSP::uiUpdate)) {
                *result = 3;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject UIStateSP::staticMetaObject = { {
    &UIState::staticMetaObject,
    qt_meta_stringdata_UIStateSP.data,
    qt_meta_data_UIStateSP,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *UIStateSP::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *UIStateSP::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_UIStateSP.stringdata0))
        return static_cast<void*>(this);
    return UIState::qt_metacast(_clname);
}

int UIStateSP::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = UIState::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 5)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 5;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 5)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 5;
    }
    return _id;
}

// SIGNAL 0
void UIStateSP::sunnylinkRoleChanged(bool _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void UIStateSP::sunnylinkRolesChanged(std::vector<RoleModel> _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void UIStateSP::sunnylinkDeviceUsersChanged(std::vector<UserModel> _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void UIStateSP::uiUpdate(const UIStateSP & _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}
struct qt_meta_stringdata_DeviceSP_t {
    QByteArrayData data[1];
    char stringdata0[9];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_DeviceSP_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_DeviceSP_t qt_meta_stringdata_DeviceSP = {
    {
QT_MOC_LITERAL(0, 0, 8) // "DeviceSP"

    },
    "DeviceSP"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_DeviceSP[] = {

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

void DeviceSP::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    Q_UNUSED(_o);
    Q_UNUSED(_id);
    Q_UNUSED(_c);
    Q_UNUSED(_a);
}

QT_INIT_METAOBJECT const QMetaObject DeviceSP::staticMetaObject = { {
    &Device::staticMetaObject,
    qt_meta_stringdata_DeviceSP.data,
    qt_meta_data_DeviceSP,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *DeviceSP::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *DeviceSP::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_DeviceSP.stringdata0))
        return static_cast<void*>(this);
    return Device::qt_metacast(_clname);
}

int DeviceSP::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = Device::qt_metacall(_c, _id, _a);
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
