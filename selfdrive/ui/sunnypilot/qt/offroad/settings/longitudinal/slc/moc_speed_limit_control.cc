/****************************************************************************
** Meta object code from reading C++ file 'speed_limit_control.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.12.8)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "speed_limit_control.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'speed_limit_control.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.12.8. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_SpeedLimitControl_t {
    QByteArrayData data[3];
    char stringdata0[44];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_SpeedLimitControl_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_SpeedLimitControl_t qt_meta_stringdata_SpeedLimitControl = {
    {
QT_MOC_LITERAL(0, 0, 17), // "SpeedLimitControl"
QT_MOC_LITERAL(1, 18, 24), // "slcSettingsButtonClicked"
QT_MOC_LITERAL(2, 43, 0) // ""

    },
    "SpeedLimitControl\0slcSettingsButtonClicked\0"
    ""
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_SpeedLimitControl[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   19,    2, 0x06 /* Public */,

 // signals: parameters
    QMetaType::Void,

       0        // eod
};

void SpeedLimitControl::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<SpeedLimitControl *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->slcSettingsButtonClicked(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (SpeedLimitControl::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&SpeedLimitControl::slcSettingsButtonClicked)) {
                *result = 0;
                return;
            }
        }
    }
    Q_UNUSED(_a);
}

QT_INIT_METAOBJECT const QMetaObject SpeedLimitControl::staticMetaObject = { {
    &ExpandableToggleRow::staticMetaObject,
    qt_meta_stringdata_SpeedLimitControl.data,
    qt_meta_data_SpeedLimitControl,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *SpeedLimitControl::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *SpeedLimitControl::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_SpeedLimitControl.stringdata0))
        return static_cast<void*>(this);
    return ExpandableToggleRow::qt_metacast(_clname);
}

int SpeedLimitControl::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = ExpandableToggleRow::qt_metacall(_c, _id, _a);
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
    return _id;
}

// SIGNAL 0
void SpeedLimitControl::slcSettingsButtonClicked()
{
    QMetaObject::activate(this, &staticMetaObject, 0, nullptr);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
